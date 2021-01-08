#!/usr/bin/python
'''
Run a BOHB search to optimize a feed-forward neural network 
architecture for classifying MCI converter vs. non-converter
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import numpy as np
import pandas as pd
import argparse
import pickle
from ray.tune.integration.keras import TuneReporterCallback
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

SEED = 3797
SPLITS = os.path.abspath('splits_20test_5inner.pkl')
EPOCHS = 100
NMODELS = 100
BATCHSIZE = 32
INNERFOLDS = 5

def train(config, checkpoint_dir=None):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL')
    from medl import tune_models # pylint: disable=import-error
    
    tf.get_logger().setLevel('ERROR')
    
    # Checkpointing for hyperband
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, 'checkpoint')) as f:
            dictState = json.loads(f.read())
            iStartEpoch = dictState['step'] + 1
    else:
        iStartEpoch = 0
        dictState = {'step': iStartEpoch
                     }

    # Load partitioned data 
    with open(SPLITS, 'rb') as f:
        dictSplits = pickle.load(f)
    
    # normalize values, then divide into batches
    lsTrainBatches = []
    lsTrainData = []
    lsValData = []

    for iInnerFold in range(INNERFOLDS):
        dfDataTrain, dfLabelsTrain = dictSplits[f'train{iInnerFold}']
        dfDataVal, dfLabelsVal = dictSplits[f'val{iInnerFold}']

        preproc = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer())])
        arrXTrain = preproc.fit_transform(dfDataTrain.astype(float))
        arrXVal = preproc.transform(dfDataVal.astype(float))
        datasetTrain = tf.data.Dataset.from_tensor_slices((arrXTrain, dfLabelsTrain.values))
        lsTrainBatches += [datasetTrain.shuffle(1024, seed=SEED).batch(BATCHSIZE)]
        lsTrainData += [(arrXTrain, dfLabelsTrain.values)]
        lsValData += [(arrXVal, dfLabelsVal.values)]

    # Create identical models for each fold
    lsModels = []
    for iInnerFold in range(INNERFOLDS):
        lsModels += [tune_models.create_classifier_from_config(config, (arrXTrain.shape[1],), nMinNeurons=4)]

    # If resuming previously Hyperband-paused trial, reload the saved weights
    if checkpoint_dir:
        for i in range(INNERFOLDS):
            lsModels[i].load_weights(os.path.join(checkpoint_dir, f'fold{i}.h5'))

    # Define the training loop
    for iEpoch in range(iStartEpoch, EPOCHS):
        lsMetrics = []
        # Train each fold's model for one epoch
        for iFold in range(INNERFOLDS):
            for arrBatchX, arrBatchY in lsTrainBatches[iFold]:
                dictMetrics = lsModels[iFold].train_on_batch(arrBatchX, arrBatchY, reset_metrics=False,
                                                             return_dict=True)        
            arrXTrain, arrYTrain = lsTrainData[iFold]
            dictMetrics = lsModels[iFold].evaluate(arrXTrain, arrYTrain, return_dict=True, verbose=0)                                                                        
            arrXVal, arrYVal = lsValData[iFold]
            dictVal = lsModels[iFold].evaluate(arrXVal, arrYVal, return_dict=True, verbose=0)
            dictMetrics.update({'val_loss': dictVal['loss'], 
                                'val_auroc': dictVal['auroc']})

            lsMetrics += [dictMetrics] 

        dfMetrics = pd.DataFrame(lsMetrics)
        # Log the mean performance over the folds to Tune
        tune.report(**dfMetrics.mean().to_dict())

        # Update the checkpoint and save latest weights
        dictState['step'] = iEpoch
        with tune.checkpoint_dir(step=iEpoch) as checkpoint_dir:
            for i in range(INNERFOLDS):
                    lsModels[i].save_weights(os.path.join(checkpoint_dir, f'fold{i}.h5'))
            with open(os.path.join(checkpoint_dir, 'checkpoint'), 'w') as f:
                f.write(json.dumps(dictState))


if __name__ == "__main__":
    import ray
    import datetime
    from ray import tune
    from ray.tune.schedulers import HyperBandForBOHB              
    from ray.tune.suggest.bohb import TuneBOHB
    from ray.tune import CLIReporter
    # import tensorflow as tf
    import ConfigSpace as CS
    from ConfigSpace import hyperparameters
    import warnings

    ray.init()
    strOutputDir = sys.argv[1]
    strTimestamp = datetime.date.today().strftime('%Y%m%d')

    configSpace = CS.ConfigurationSpace()
    # Global model hyperparams
    configSpace.add_hyperparameter(hyperparameters.CategoricalHyperparameter('batchnorm', [False, True]))
    configSpace.add_hyperparameter(hyperparameters.CategoricalHyperparameter('activation', ['ELU', 'LeakyReLU', 'ReLU', 'PReLU']))
    configSpace.add_hyperparameter(hyperparameters.UniformIntegerHyperparameter('hiddenlayers', lower=1, upper=5))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('l1', lower=0.05, upper=0.9))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('l2', lower=0.05, upper=0.9))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('dropout', lower=0.2, upper=0.8))
    # Nadam hyperparams
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('lr', lower=0.001, upper=0.03))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('beta_1', lower=0.3, upper=0.9))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('beta_2', lower=0.6, upper=0.9))
    # Dense layer 0 
    configSpace.add_hyperparameter(hyperparameters.UniformIntegerHyperparameter('dense0_neurons', lower=8, upper=128))
    # Taper rate (decrease in neurons for subsequent hidden layers)
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('taper', lower=0.05, upper=0.5))

    scheduler = HyperBandForBOHB(time_attr='training_iteration',
                                 metric='val_auroc',
                                 mode='max',
                                 max_t=EPOCHS)
    search = TuneBOHB(configSpace, max_concurrent=2, metric='val_auroc', mode='max')       

    # Have the command line output report train and val performance
    reporter = CLIReporter(metric_columns=['training_iteration', 'auroc', 'val_auroc'],
                           parameter_columns=['hiddenlayers', 'dense0_neurons', 'taper', 'lr'],
                           max_report_frequency=30)

    os.makedirs(strOutputDir, exist_ok=True)
    tune.run(train,
             name=strTimestamp,
             scheduler=scheduler,
             search_alg=search,
             stop={"training_iteration": EPOCHS}, # Stop when 100 epochs are reached
             num_samples=NMODELS,
             resources_per_trial={'cpu': 20, 'gpu': 1},
             local_dir=strOutputDir,
             progress_reporter=reporter)
