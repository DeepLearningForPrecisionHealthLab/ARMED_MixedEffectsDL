'''
After running BOHB HPO, evaluate the best performing model configurations on test data.
'''

RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_3d_LOSO_20210210/me_unet_hpo'
SPLITS = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d/LOSO_splits.pkl'
SEED = 493
IMAGESHAPE = (64, 64, 32, 1)
BATCH = 12

import os
import sys
import pickle
import glob
import pandas as pd
import tensorflow as tf
from ray.tune.analysis import Analysis
sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL')
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL/prostate_segmentation')
from prostate_datagenerator import SegmentationDataFrameGenerator3D # pylint: disable=import-error
from candidate_architectures.random_slope_int_l1norm import me_unet3d

with open(SPLITS, 'rb') as f:
    dictSplits = pickle.load(f)

# List of directories containing HPO results for each held-out site
lsSiteDirs = sorted(glob.glob(os.path.join(RESULTSDIR, '*')))

dfResults = pd.DataFrame(columns=['train', 'val', 'test', 'held-out site'])
for strSiteDir in lsSiteDirs:
    strSite = strSiteDir.split(os.path.sep)[-1]

    train_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['train'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                seed=SEED)
    nGroups = train_data.group_encoder.categories_[0].size
    val_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['val'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                group_encoding=train_data.group_encoder.categories_[0],
                                                seed=SEED)
    test_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['test'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                group_encoding=train_data.group_encoder.categories_[0],
                                                seed=SEED)
    loso_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['held-out site'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                group_encoding=train_data.group_encoder.categories_[0],
                                                seed=SEED)

    analysis = Analysis(strSiteDir)
    strBestTrialDir = analysis.get_best_logdir('val_dice', 'max')
    config = analysis.get_best_config('val_dice', 'max')

    model = me_unet3d(IMAGESHAPE, nGroups,
                    loc_init_range=config['loc_init_range'],
                    scale_init_min=config['scale_init_min'],
                    scale_init_max=config['scale_init_min'] + config['scale_init_range'],
                    prior_scale=config['prior_scale'],
                    l1norm_weight=config['l1norm_weight'])
    # model.load_weights(os.path.join(strBestTrialDir, 'best_weights.h5'))                
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=config['learning_rate']), 
                loss=dice_bce_loss,
                loss_weights=[config['seg_loss_weight']],
                metrics=[dice])
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max')]

    model.fit(train_data,
            validation_data=val_data,
            epochs=200,
            steps_per_epoch=len(train_data),
            validation_steps=len(val_data),
            callbacks=lsCallbacks,
            verbose=1)


    fDiceTrain = model.evaluate(train_data)[1]
    fDiceVal = model.evaluate(val_data)[1]
    fDiceTest = model.evaluate(test_data)[1]
    fDiceLOSO = model.evaluate(loso_data)[1]
    dfResults.loc[strSite] = [fDiceTrain, fDiceVal, fDiceTest, fDiceLOSO]
print(dfResults)