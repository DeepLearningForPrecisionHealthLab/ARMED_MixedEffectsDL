'''
Evaluate the trained ME-Unet, which was trained with transfered weights from the original Unet, on the held-out test and HK site data.
'''

DATADIR = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices'
WEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_me_20210110/random_slope_featurewise_transfer/val/final_weights.h5' 
SEED = 493
IMAGESHAPE = (384, 384, 1)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
sys.path.append('./candidate_architectures')
from random_slope_featurewise_hpo import unet # pylint: disable=import-error

# Create data generators
test_data = SegmentationDataGenerator(os.path.join(DATADIR, 'test', 'image'),
                                        os.path.join(DATADIR, 'test', 'mask'),
                                        IMAGESHAPE,
                                        batch_size=10,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        return_group=True,
                                        seed=SEED)
hk_data = SegmentationDataGenerator(os.path.join(DATADIR, 'hk', 'image'),
                                        os.path.join(DATADIR, 'hk', 'mask'),
                                        IMAGESHAPE,
                                        batch_size=10,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        return_group=True,
                                        seed=SEED)      
hk_data.groups = np.zeros((len(hk_data.images), 5)) # dummy design matrix 

model = unet(IMAGESHAPE, 5, 
            {'re_int_loc_range': 1, 
            're_int_std_min': 0.1, 
            're_int_std_range': 0.1, 
            'prior_std': 0.1,
            'learning_rate': 1e-3,
            'seg_loss_weight': 1000})
model.load_weights(WEIGHTS)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
            loss=dice_bce_loss, 
            metrics=[dice])
print('Test performance')
model.evaluate(test_data)
print('HK site performance')
model.evaluate(hk_data)