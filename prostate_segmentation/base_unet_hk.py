'''
Evaluate the trained basic Unet on the held-out HK site.
'''
DATADIR = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_20210110'
WEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_20210110/test/final_weights.h5'
SEED = 493
IMAGESHAPE = (384, 384, 1)

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
sys.path.append('../')
from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
import albumentations
from prostate_datagenerator import SegmentationDataGenerator # pylint: disable=import-error


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))                               

# Create data generators
# train_data = SegmentationDataGenerator(os.path.join(DATADIR, 'train', 'image'),
#                                         os.path.join(DATADIR, 'train', 'mask'),
#                                         IMAGESHAPE,
#                                         samplewise_center=True,
#                                         samplewise_std_normalization=True,
#                                         seed=SEED)
val_data = SegmentationDataGenerator(os.path.join(DATADIR, 'hk', 'image'),
                                        os.path.join(DATADIR, 'hk', 'mask'),
                                        IMAGESHAPE,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        seed=SEED)

model = unet.unet(input_size=IMAGESHAPE, pretrained_weights=WEIGHTS)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
            loss=dice_bce_loss, 
            metrics=[dice])

model.evaluate(val_data)