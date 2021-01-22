DATADIR = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_me_20210110/random_slope_featurewise/bohb'
SEED = 493
IMAGESHAPE = (384, 384, 1)
EPOCHS = 100
NMODELS = 30

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import albumentations

def train(config, checkpoint_dir=None):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf    
    import tensorflow.keras.layers as tkl
    sys.path.append(os.path.abspath('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL'))
    from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
    from prostate_datagenerator import SegmentationDataGenerator # pylint: disable=import-error
    from candidate_architectures.random_slope_featurewise_hpo import unet
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

    # pylint: disable=no-member
    augmentations = albumentations.Compose([albumentations.VerticalFlip(p=0.5),
                                            albumentations.HorizontalFlip(p=0.5),
                                            albumentations.ShiftScaleRotate(
                                                shift_limit=0.1, 
                                                scale_limit=0.1, 
                                                rotate_limit=20, 
                                                interpolation=cv2.INTER_CUBIC,
                                                border_mode=cv2.BORDER_CONSTANT,
                                                p=0.5)
                                            ])

    # Create data generators
    train_data = SegmentationDataGenerator(os.path.join(DATADIR, 'train', 'image'),
                                           os.path.join(DATADIR, 'train', 'mask'),
                                           IMAGESHAPE,
                                           batch_size=16,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True,
                                           augmentation=augmentations,
                                           return_group=True,
                                           seed=SEED)
    val_data = SegmentationDataGenerator(os.path.join(DATADIR, 'val', 'image'),
                                           os.path.join(DATADIR, 'val', 'mask'),
                                           IMAGESHAPE,
                                           batch_size=16,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True,
                                           return_group=True,
                                           seed=SEED)

    # Create model with sampled hyperparams
    model = unet(IMAGESHAPE, 5, config)

    # If resuming previously Hyperband-paused trial, reload the saved weights
    if checkpoint_dir:
        model.load_weights(os.path.join(checkpoint_dir, f'weights.h5'))

    # Define the training loop
    for iEpoch in range(iStartEpoch, EPOCHS):
        lsMetrics = []
        for arrBatchX, arrBatchY in train_data:
            dictMetrics = model.train_on_batch(arrBatchX, arrBatchY, reset_metrics=False, return_dict=True)        
        dictMetrics = model.evaluate(train_data, return_dict=True, verbose=0)                                                                        
        dictVal = model.evaluate(val_data, return_dict=True, verbose=0)
        dictMetrics.update({'val_loss': dictVal['loss'], 
                            'val_dice': dictVal['dice']})

        lsMetrics += [dictMetrics] 

        dfMetrics = pd.DataFrame(lsMetrics)
        # Log the mean performance over the folds to Tune
        tune.report(**dfMetrics.mean().to_dict())

        # Update the checkpoint and save latest weights
        dictState['step'] = iEpoch
        with tune.checkpoint_dir(step=iEpoch) as checkpoint_dir:
            model.save_weights(os.path.join(checkpoint_dir, f'weights.h5'))
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
    strTimestamp = datetime.date.today().strftime('%Y%m%d')

    configSpace = CS.ConfigurationSpace()

    # Random effect hyperparams
    # S.d. of the random normal initialization of the RE intercepts
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('re_int_loc_init', lower=0.01, upper=0.1))
    # Min and range of the random uniform initialization of the RE variance params
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('re_int_std_min', lower=1.0, upper=2.0))
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('re_int_std_range', lower=0.05, upper=0.2))
    # S.d. of the normal prior distribution
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('prior_std', lower=1.0, upper=2.0))

    # Nadam hyperparams
    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('learning_rate', lower=1e-3, upper=3e-2))

    configSpace.add_hyperparameter(hyperparameters.UniformFloatHyperparameter('seg_loss_weight', lower=50, upper=200))
    
    scheduler = HyperBandForBOHB(time_attr='training_iteration',
                                 metric='val_loss',
                                 mode='min',
                                 max_t=EPOCHS)
    search = TuneBOHB(configSpace, max_concurrent=2, metric='val_loss', mode='min')       

    # Have the command line output report train and val performance
    reporter = CLIReporter(metric_columns=['training_iteration', 'dice', 'val_dice', 'val_loss'],
                           parameter_columns=['re_int_loc_init', 're_int_std_min', 're_int_std_range', 'prior_std'],
                           max_report_frequency=30)

    os.makedirs(RESULTSDIR, exist_ok=True)
    tune.run(train,
             name=strTimestamp,
             scheduler=scheduler,
             search_alg=search,
             stop={"training_iteration": EPOCHS}, # Stop when max epochs are reached
             num_samples=NMODELS,
             resources_per_trial={'cpu': 20, 'gpu': 1},
             local_dir=RESULTSDIR,
             progress_reporter=reporter)



    
