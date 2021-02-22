'''
Train classic U-Net to segment the prostate from T2-weighted MRIs
'''
## 128 x 128 x 32 images (0.75 mm isotropic voxels)
SPLITS = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d_128/LOSO_splits.pkl'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_3d_128_LOSO_20210219/base_unet'
SEED = 493
IMAGESHAPE = (128, 128, 32, 1)
BATCH = 4

## 64 x 64 x 32 images (2 mm isotropic voxels)
# RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_3d_LOSO_20210210/base_unet'
# SPLITS = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d/LOSO_splits.pkl'
# SEED = 493
# IMAGESHAPE = (64, 64, 32, 1)
# BATCH = 12

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

def train(strSite):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices()
    print(gpus)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    strOutputDir = os.path.join(RESULTSDIR, strSite)
    os.makedirs(strOutputDir, exist_ok=True)

    # Create data generators
    with open(SPLITS, 'rb') as f:
        dictSplits = pickle.load(f)
    
    train_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['train'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)
    val_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['val'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)
    test_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['test'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)                                                
    loso_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['held-out site'],
                                                IMAGESHAPE,
                                                batch_size=BATCH,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)                                                

    model = unet.unet3d(input_size=IMAGESHAPE)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
                loss=dice_bce_loss, 
                metrics=[dice])

    strLogDir = os.path.join(strOutputDir, 'logs')
    os.makedirs(strLogDir, exist_ok=True)
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=10, restore_best_weights=True, mode='max'),
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
    fDiceTest = model.evaluate(test_data)[1]
    fDiceLOSO = model.evaluate(loso_data)[1]

    with open(os.path.join(strOutputDir, 'results.txt'), 'w') as f:
        f.write(f'Train: {fDiceTrain}\nVal: {fDiceVal}\nTest: {fDiceTest}\n{strSite}: {fDiceLOSO}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--site', '-s', type=str, help='Site to hold out of training')
    arguments = args.parse_args()
    train(arguments.site)