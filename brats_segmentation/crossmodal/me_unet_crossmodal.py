'''
Train random slopee U-Net to segment GBMs agnostic to MRI contrast.
'''
DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020_separatecontrasts'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/gbm_segmentation_crossmodal_me_20210110'
SEED = 399
IMAGESHAPE = (240, 240, 1)

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
sys.path.append('../candidate_architectures')
sys.path.append('../../')
# from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
from medl.datagenerator import SegmentationDataGenerator # pylint: disable=import-error
from random_slope_featurewise import unet_mixedeffects
import albumentations

def train(mode='val', augment=True):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices()
    print(gpus)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    if mode == 'val':
        strOutputDir = os.path.join(RESULTSDIR, 'val')
    elif mode == 'test':
        strOutputDir = os.path.join(RESULTSDIR, 'test')
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
    train_data = SegmentationDataGenerator(os.path.join(DATADIR, 'train', 'image'),
                                           os.path.join(DATADIR, 'train', 'mask'),
                                           IMAGESHAPE,
                                           return_contrast=True,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True,
                                           augmentation=augmentations,
                                           seed=SEED)
    val_data = SegmentationDataGenerator(os.path.join(DATADIR, mode, 'image'),
                                           os.path.join(DATADIR, mode, 'mask'),
                                           IMAGESHAPE,
                                           return_contrast=True,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True,
                                           seed=SEED)

    model = unet_mixedeffects((4,), input_size=IMAGESHAPE)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
                loss=dice_bce_loss, 
                metrics=[dice])
    # arrValBatchData, arrValBatchLabel = val_data[0] # example batch for creating segmentation result figures
    strLogDir = os.path.join(strOutputDir, 'logs')
    os.makedirs(strLogDir, exist_ok=True)
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max'),
                   tf.keras.callbacks.CSVLogger(os.path.join(strOutputDir, 'training.log')),
                   tf.keras.callbacks.TensorBoard(log_dir=strLogDir)]

    model.fit(train_data,
            validation_data=val_data,
            epochs=500,
            batch_size=32,
            steps_per_epoch=len(train_data),
            validation_steps=len(val_data),
            callbacks=lsCallbacks,
            verbose=1)
    model.save_weights(os.path.join(strOutputDir, 'final_weights.h5'))                    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', '-m', default='val', help='Evalute performance on val or test')
    args.add_argument('--augment', '-a', default=True, type=bool, help='True/False whether to apply augmentation')
    arguments = args.parse_args()
    train(arguments.mode, arguments.augment)
