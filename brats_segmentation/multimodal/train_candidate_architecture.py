'''
Train U-Net with embedded random effect layer to segment GBMs from multimodal MRI axial slices. 
Random intercepts are fitted for each source subject (as the training data contains multiple axial 
slices per subject)

Applies data augmentation using albumentations and custom Keras data generator from 
https://github.com/mjkvaak/ImageDataAugmentor 

Usage:
python train_candidate_architecture.py -f <fold> -o <output_dir> -m candidate_architectures.module_with_model_fn
'''
DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020_groups'
SEED = 9786

import os
from random import sample
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import albumentations
import matplotlib.pyplot as plt
import importlib

sys.path.append('../../')
from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice, dice_bce_loss # pylint: disable=import-error
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
from medl.datagenerator import GroupedDataGenerator # pylint: disable=import-error

def train(fold, strResultsDir, strModelPath, augment=True):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices()
    print(gpus)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    module = importlib.import_module(strModelPath)

    if fold == 'test':
        strFoldDir = os.path.join(strResultsDir, f'final_test')
        strDataPath = os.path.join(DATADIR, 'train_test.npz')
        # Load partitioned data
        dictData = np.load(strDataPath)
        arrValData = np.concatenate((dictData['t1_test'], dictData['t1ce_test'], dictData['t2_test'], dictData['flair_test']), axis=-1)
        arrValLabel = dictData['mask_test']
        arrValGroups = dictData['subject_test'].reshape(-1, 1)
    else:
        strFoldDir = os.path.join(strResultsDir, f'fold{fold}')
        strDataPath = os.path.join(DATADIR, f'fold{fold}.npz')
        # Load partitioned data
        dictData = np.load(strDataPath)
        arrValData = np.concatenate((dictData['t1_val'], dictData['t1ce_val'], dictData['t2_val'], dictData['flair_val']), axis=-1)
        arrValLabel = dictData['mask_val']
        arrValGroups = dictData['subject_val'].reshape(-1, 1)
    
    arrTrainData = np.concatenate((dictData['t1_train'], dictData['t1ce_train'], dictData['t2_train'], dictData['flair_train']), axis=-1)
    arrTrainLabel = dictData['mask_train']
    arrTrainGroups = dictData['subject_train'].reshape(-1, 1)
    del dictData
    os.makedirs(strFoldDir, exist_ok=True)

    tupImgShape = arrTrainData.shape[1:]
    print('Images are size', tupImgShape)

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
    train_gen = GroupedDataGenerator(arrTrainData, arrTrainLabel, arrTrainGroups,
                                    label_type='mask',
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    augmentation=augmentations,
                                    seed=SEED)
    val_gen = GroupedDataGenerator(arrValData, arrValLabel, arrValGroups,
                                    label_type='mask',
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    seed=SEED)
    nGroups = np.concatenate(train_gen.group_encoder.categories_).shape[0]
    val_gen.set_dummy_encoder((nGroups,))                                    

    model = module.unet_mixedeffects((nGroups,), input_size=tupImgShape, random_int=True)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
                loss=dice_bce_loss, 
                metrics=[dice])
    # Grab an example batch to use for visualizing predictions
    arrValBatchData, arrValBatchLabel = val_gen[0]
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max'),
                SaveImagesCallback(arrValBatchData, arrValBatchLabel, strFoldDir, random_effects=True),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(strFoldDir, 'weights.{epoch:03d}.hdf5'),
                                                    monitor='val_dice',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    save_freq=5 * len(train_gen)),
                tf.keras.callbacks.CSVLogger(os.path.join(strFoldDir, 'training.log'))]

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=500,
                        batch_size=32,
                        steps_per_epoch=len(train_gen),
                        validation_steps=len(val_gen),
                        callbacks=lsCallbacks,
                        verbose=1)
    model.save_weights(os.path.join(strFoldDir, 'final_weights.h5'))                    
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(strFoldDir, 'history.csv'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--fold', '-f', default='test', help='Cross-validation fold to run (int) or "test" to train on final train/test split')
    args.add_argument('--output', '-o', required=True, help='Directory for saving outputs')
    args.add_argument('--model', '-m', required=True, help='Name of module containing the candidate architecture')
    args.add_argument('--augment', '-a', default=True, type=bool, help='True/False whether to apply augmentation')
    arguments = args.parse_args()
    train(arguments.fold, arguments.output, arguments.model, arguments.augment)