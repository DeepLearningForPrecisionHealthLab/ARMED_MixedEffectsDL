'''
Train classic U-Net to segment the prostate from T2-weighted MRIs
'''
SPLITS = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/LOSO_splits.pkl'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_LOSO/base_unet'
SEED = 493
IMAGESHAPE = (384, 384, 1)

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
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
import albumentations
from prostate_datagenerator import SegmentationDataFrameGenerator # pylint: disable=import-error

def train(strSite, augment=True):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices()
    print(gpus)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    strOutputDir = os.path.join(RESULTSDIR, strSite)
    os.makedirs(strOutputDir, exist_ok=True)

    # Define augmentation pipeline
    if augment:
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
    else:
        augmentations = None                                                

    # Create data generators
    with open(SPLITS, 'rb') as f:
        dictSplits = pickle.load(f)
    
    train_data = SegmentationDataFrameGenerator(dictSplits[strSite]['train'],
                                                IMAGESHAPE,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                augmentation=augmentations,
                                                seed=SEED)
    val_data = SegmentationDataFrameGenerator(dictSplits[strSite]['val'],
                                                IMAGESHAPE,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)
    test_data = SegmentationDataFrameGenerator(dictSplits[strSite]['test'],
                                                IMAGESHAPE,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)                                                
    loso_data = SegmentationDataFrameGenerator(dictSplits[strSite]['held-out site'],
                                                IMAGESHAPE,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)                                                

    model = unet.unet(input_size=IMAGESHAPE)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
                loss=dice_bce_loss, 
                metrics=[dice])

    strLogDir = os.path.join(strOutputDir, 'logs')
    os.makedirs(strLogDir, exist_ok=True)
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=10, restore_best_weights=True, mode='max'),
                   tf.keras.callbacks.CSVLogger(os.path.join(strOutputDir, 'training.log')),
                   SaveImagesCallback(val_data[0][0], val_data[0][1], strOutputDir),
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
    fDiceTest = model.evaluate(test_data)[1]
    fDiceLOSO = model.evaluate(loso_data)[1]

    with open(os.path.join(strOutputDir, 'results.txt'), 'w') as f:
        f.write(f'Train: {fDiceTrain}\nTest: {fDiceTest}\n{strSite}: {fDiceLOSO}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--site', '-s', type=str, help='Site to hold out of training')
    args.add_argument('--augment', '-a', default=True, type=bool, help='True/False whether to apply augmentation')
    arguments = args.parse_args()
    train(arguments.site, arguments.augment)
