'''
Train classic U-Net to segment the prostate from T2-weighted MRIs
'''
SPLITS = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d/QIN_external.pkl'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_3d_LOSO_20210210/base_unet'
SEED = 493
IMAGESHAPE = (64, 64, 32, 1)

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pickle
import cv2
sys.path.append('../')
from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
# from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
from prostate_datagenerator import SegmentationDataFrameGenerator3D # pylint: disable=import-error

strOutputDir = os.path.join(RESULTSDIR, 'QIN_external')
os.makedirs(strOutputDir, exist_ok=True)

# Create data generators
with open(SPLITS, 'rb') as f:
    dictSplits = pickle.load(f)

train_data = SegmentationDataFrameGenerator3D(dictSplits['train'],
                                            IMAGESHAPE,
                                            batch_size=16,
                                            samplewise_center=True,
                                            samplewise_std_normalization=True,
                                            seed=SEED)
val_data = SegmentationDataFrameGenerator3D(dictSplits['val'],
                                            IMAGESHAPE,
                                            batch_size=16,
                                            samplewise_center=True,
                                            samplewise_std_normalization=True,
                                            seed=SEED)     
external_data = SegmentationDataFrameGenerator3D(dictSplits['external'],
                                                IMAGESHAPE,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)                                                                                         

model = unet.unet3d(input_size=IMAGESHAPE)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
            loss=dice_bce_loss, 
            metrics=[dice])

strLogDir = os.path.join(strOutputDir, 'logs')
os.makedirs(strLogDir, exist_ok=True)
lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max'),
                tf.keras.callbacks.CSVLogger(os.path.join(strOutputDir, 'training.log')),
            #    SaveImagesCallback(val_data[0][0], val_data[0][1], strOutputDir),
                tf.keras.callbacks.TensorBoard(log_dir=strLogDir)]

model.fit(train_data,
        validation_data=val_data,
        epochs=200,
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data),
        callbacks=lsCallbacks,
        verbose=1)
model.save_weights(os.path.join(strOutputDir, 'final_weights.h5'))                    

fDiceTrain = model.evaluate(train_data)[1]
fDiceVal = model.evaluate(val_data)[1]
fDiceExternal = model.evaluate(external_data)[1]

with open(os.path.join(strOutputDir, 'results.txt'), 'w') as f:
    f.write(f'Train: {fDiceTrain}\nVal: {fDiceVal}\nExternal: {fDiceExternal}')
