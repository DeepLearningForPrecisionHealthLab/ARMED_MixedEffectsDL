'''
Train classic U-Net to segment GBMs from multimodal MRI axial slices. 

Applies data augmentation using albumentations and custom Keras data generator from 
https://github.com/mjkvaak/ImageDataAugmentor 
'''
DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/gbm_segmentation_augmentation_20210105'
SEED = 9786

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import albumentations
import matplotlib.pyplot as plt

sys.path.append('../../')
from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice, dice_bce_loss # pylint: disable=import-error
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
sys.path.append('../../external/ImageDataAugmentor')
from ImageDataAugmentor import image_data_augmentor # pylint: disable=import-error

def train(fold, augment=True):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices()
    print(gpus)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    if fold == 'test':
        strFoldDir = os.path.join(RESULTSDIR, f'final_test')
        strDataPath = os.path.join(DATADIR, 'train_test.npz')
        # Load partitioned data
        dictData = np.load(strDataPath)
        arrValData = np.concatenate((dictData['t1_test'], dictData['t1ce_test'], dictData['t2_test'], dictData['flair_test']), axis=-1)
        arrValLabel = dictData['mask_test']
    else:
        strFoldDir = os.path.join(RESULTSDIR, f'fold{fold}')
        strDataPath = os.path.join(DATADIR, f'fold{fold}.npz')
        # Load partitioned data
        dictData = np.load(strDataPath)
        arrValData = np.concatenate((dictData['t1_val'], dictData['t1ce_val'], dictData['t2_val'], dictData['flair_val']), axis=-1)
        arrValLabel = dictData['mask_val']
    
    arrTrainData = np.concatenate((dictData['t1_train'], dictData['t1ce_train'], dictData['t2_train'], dictData['flair_train']), axis=-1)
    arrTrainLabel = dictData['mask_train']
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
    img_gen = image_data_augmentor.ImageDataAugmentor(augment=augmentations, 
                                                    samplewise_center=True,
                                                    samplewise_std_normalization=True,
                                                    input_augment_mode='image',
                                                    seed=SEED)
    mask_gen = image_data_augmentor.ImageDataAugmentor(augment=augmentations, 
                                                    input_augment_mode='mask', 
                                                    seed=SEED)  
    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)   

    train_data = img_gen.flow(arrTrainData.astype(np.float32), shuffle=True)
    train_label = mask_gen.flow(arrTrainLabel.astype(np.float32), shuffle=True)
    train_iterator = zip(train_data, train_label)
    val_iterator = val_gen.flow(arrValData.astype(np.float32), arrValLabel.astype(np.float32))

    model = unet.unet(input_size=tupImgShape)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
                loss=dice_bce_loss, 
                metrics=[dice])
    arrValBatchData, arrValBatchLabel = next(val_iterator)
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max'),
                SaveImagesCallback(arrValBatchData, arrValBatchLabel, strFoldDir),
                tf.keras.callbacks.ModelCheckpoint(os.path.join(strFoldDir, 'weights.{epoch:03d}.hdf5'),
                                                    monitor='val_dice',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    save_freq=5 * arrTrainData.shape[0] // 32),
                tf.keras.callbacks.CSVLogger(os.path.join(strFoldDir, 'training.log'))]
    nTrainSamples = arrTrainData.shape[0]
    nValSamples = arrValData.shape[0]
    history = model.fit(train_iterator,
                        validation_data=val_iterator,
                        epochs=500,
                        batch_size=32,
                        steps_per_epoch=np.ceil(nTrainSamples / 32),
                        validation_steps=np.ceil(nValSamples / 32),
                        callbacks=lsCallbacks,
                        verbose=1)
    model.save_weights(os.path.join(strFoldDir, 'final_weights.h5'))                    
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(strFoldDir, 'history.csv'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--fold', '-f', default='test', help='Cross-validation fold to run (int) or "test" to train on final train/test split')
    args.add_argument('--augment', '-a', default=True, type=bool, help='True/False whether to apply augmentation')
    arguments = args.parse_args()
    train(arguments.fold, arguments.augment)