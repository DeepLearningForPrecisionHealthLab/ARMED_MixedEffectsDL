SPLITS = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d/LOSO_splits.pkl'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_3d_LOSO_20210210/me_unet_hpo'
SEED = 493
IMAGESHAPE = (64, 64, 32, 1)
EPOCHS = 200 # max epochs per model
NMODELS = 100 # search iterations

import os
import sys
import argparse
import json
import pandas as pd

def train(config, checkpoint_dir=None):

    import tensorflow as tf
    import numpy as np
    import pickle
    sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL')
    from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
    # from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
    sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL/prostate_segmentation')
    from prostate_datagenerator import SegmentationDataFrameGenerator3D # pylint: disable=import-error
    from candidate_architectures.random_slope_int_l1norm import me_unet3d
    tf.get_logger().setLevel('ERROR')
    
    # Checkpointing for hyperband
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, 'checkpoint.json'), 'r') as f:
            dictState = json.loads(f.read())
            iStartEpoch = dictState['step'] + 1
    else:
        iStartEpoch = 0
        dictState = {'step': iStartEpoch,
                     'best_val_dice': 0
                     }

    # Create data generators
    with open(SPLITS, 'rb') as f:
        dictSplits = pickle.load(f)
    strSite = config['held-out_site']
    # Batch size of 16 is too large for the ME model (though it was fine for the base model)
    train_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['train'],
                                                IMAGESHAPE,
                                                batch_size=12,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                seed=SEED)
    nGroups = train_data.group_encoder.categories_[0].size
    val_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['val'],
                                                IMAGESHAPE,
                                                batch_size=12,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                group_encoding=train_data.group_encoder.categories_[0],
                                                seed=SEED)

    # Create model with sampled hyperparams
    model = me_unet3d(IMAGESHAPE, nGroups,
                      loc_init_range=config['loc_init_range'],
                      scale_init_min=config['scale_init_min'],
                      scale_init_max=config['scale_init_min'] + config['scale_init_range'],
                      prior_scale=config['prior_scale'],
                      l1norm_weight=config['l1norm_weight'])
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=config['learning_rate']), 
                  loss=dice_bce_loss,
                  loss_weights=[config['seg_loss_weight']],
                  metrics=[dice])


    # If resuming previously Hyperband-paused trial, reload the saved weights
    if checkpoint_dir:
        strModelDir = tune.get_trial_dir()
        model.load_weights(os.path.join(strModelDir, f'checkpoint_weights.h5'))

    # Define the training loop
    for iEpoch in range(iStartEpoch, EPOCHS):
        for (arrBatchX, arrBatchGroup), arrBatchY in train_data:
            dictMetrics = model.train_on_batch((arrBatchX, arrBatchGroup), arrBatchY, reset_metrics=False, return_dict=True)        
        dictMetrics = model.evaluate(train_data, return_dict=True, verbose=0)                                                                        
        dictVal = model.evaluate(val_data, return_dict=True, verbose=0)
        dictMetrics.update({'val_loss': dictVal['loss'], 
                            'val_dice': dictVal['dice']})

        # Update the checkpoint and save latest weights
        dictState['step'] = iEpoch
        # Save most recent weights
        model.save_weights(os.path.join(tune.get_trial_dir(), f'checkpoint_weights.h5'))
        # Save "best weights" if val dice has improved
        if dictMetrics['val_dice'] > dictState['best_val_dice']:
            dictState['best_val_dice'] = dictMetrics['val_dice']
            model.save_weights(os.path.join(tune.get_trial_dir(), f'best_weights.h5'))
        with tune.checkpoint_dir(step=iEpoch) as checkpoint_dir:
            # Save training progress
            with open(os.path.join(checkpoint_dir, 'checkpoint.json'), 'w') as f:
                f.write(json.dumps(dictState))

        tune.report(**dictMetrics)

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

    args = argparse.ArgumentParser()
    args.add_argument('--site', '-s', type=str, help='Site to hold out of training')
    arguments = args.parse_args()

    ray.init()
    # strTimestamp = datetime.date.today().strftime('%Y%m%d')

    configSpace = CS.ConfigurationSpace()
    configSpace.add_hyperparameter(hyperparameters.Constant('held-out_site', arguments.site))

    # Random effect hyperparams
    # Range of the random uniform initialization of the RE intercepts
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('loc_init_range', lower=0.01, upper=1.))
    # Min and range of the random uniform initialization of the RE variance params
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('scale_init_min', lower=0.01, upper=0.1))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('scale_init_range', lower=0.05, upper=0.2))
    # S.d. of the normal prior distribution
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('prior_scale', lower=0.01, upper=1.0))

    # Nadam hyperparams
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=4e-4))

    # Loss weights
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('l1norm_weight', lower=1e-3, upper=0.1))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('seg_loss_weight', lower=1, upper=1000, log=True))
    
    scheduler = HyperBandForBOHB(time_attr='training_iteration',
                                 metric='val_dice',
                                 mode='max',
                                 max_t=EPOCHS)
    search = TuneBOHB(configSpace, max_concurrent=2, metric='val_dice', mode='max')       

    # Have the command line output report train and val performance
    reporter = CLIReporter(metric_columns=['training_iteration', 'dice', 'val_dice', 'val_loss'],
                           parameter_columns=['loc_init_range', 'seg_loss_weight', 'prior_scale', 'l1norm_weight'],
                           max_report_frequency=30)

    os.makedirs(RESULTSDIR, exist_ok=True)
    tune.run(train,
             name=arguments.site,
             scheduler=scheduler,
             search_alg=search,
             stop={"training_iteration": EPOCHS}, # Stop when max epochs are reached
             num_samples=NMODELS,
             resources_per_trial={'cpu': 20, 'gpu': 1},
             local_dir=RESULTSDIR,
             progress_reporter=reporter)
