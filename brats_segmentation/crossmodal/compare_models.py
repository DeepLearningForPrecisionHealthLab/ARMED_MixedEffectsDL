'''
Compare the performance of the base U-NET vs. the ME-UNET with a paired t-test of the Dice scores. 
'''

DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020_separatecontrasts'
UNET_WEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/gbm_segmentation_crossmodal_20210110/val/final_weights.h5'
MEUNET_WEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/gbm_segmentation_crossmodal_me_20210110/val/final_weights.h5'
SEED = 399
IMAGESHAPE = (240, 240, 1)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.stats
sys.path.append('../../')
from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
from medl.datagenerator import SegmentationDataGenerator # pylint: disable=import-error
sys.path.append('../candidate_architectures')
from random_slope_featurewise import unet_mixedeffects # pylint: disable=import-error

def dice_coef(yTrue, yPred):
    yPredMask = (yPred >= 0.5).astype(np.float)
    intersection = np.sum(yTrue * yPredMask, axis=(1, 2, 3))
    total = np.sum(yTrue, axis=(1, 2, 3)) + np.sum(yPredMask, axis=(1, 2, 3))
    return 2 * intersection / total

val_data = SegmentationDataGenerator(os.path.join(DATADIR, 'val', 'image'),
                                    os.path.join(DATADIR, 'val', 'mask'),
                                    IMAGESHAPE,
                                    return_contrast=True,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    seed=SEED)

arrInputTest = np.concatenate([t[0][0] for t in val_data], axis=0)                
arrGroupTest = np.concatenate([t[0][1] for t in val_data], axis=0)
arrSegTest = np.concatenate([t[1] for t in val_data], axis=0)

model = unet.unet(input_size=IMAGESHAPE)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
            loss=dice_bce_loss, 
            metrics=[dice])
model.load_weights(UNET_WEIGHTS)            
arrPred = model.predict(arrInputTest)
arrDice = dice_coef(arrSegTest, arrPred)

del model
modelME = unet_mixedeffects((4,), input_size=IMAGESHAPE)
modelME.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
                loss=dice_bce_loss, 
                metrics=[dice])
modelME.load_weights(MEUNET_WEIGHTS)                
arrPredME = modelME.predict((arrInputTest, arrGroupTest))                
arrDiceME = dice_coef(arrSegTest, arrPredME)

print(f'Mean Dice: base U-NET {arrDice.mean():.03f}, ME-UNET {arrDiceME.mean():.03f}')
t, p = scipy.stats.ttest_rel(arrDiceME, arrDice)
print(f't = {t:.03f}, p = {p:.05f}')