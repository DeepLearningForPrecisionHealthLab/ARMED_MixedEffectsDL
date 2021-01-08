'''
Compare the performance of the mixed effects vs. fixed effects models based on Dice scores on test data.
------
1/6/2021
'''
DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020_groups'
UNETWEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/gbm_segmentation_augmentation_20210105/final_test/final_weights.h5'
MEWEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/gbm_segmentation_mixedeffects_20210105/final_test/final_weights.h5'

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf
import scipy.stats
import matplotlib.pyplot as plt
sys.path.append('../../')
# pylint: disable=import-error
from medl.models import unet 
from medl.datagenerator import GroupedDataGenerator

strDataPath = os.path.join(DATADIR, 'train_test.npz')
# Load partitioned data
dictData = np.load(strDataPath)
arrTestData = np.concatenate((dictData['t1_test'], dictData['t1ce_test'], dictData['t2_test'], dictData['flair_test']), axis=-1)
arrTestLabel = dictData['mask_test']
arrTestGroups = dictData['subject_test'].reshape(-1, 1)
nTrainGroups = np.unique(dictData['subject_train']).shape[0]
tupImgShape = arrTestData.shape[1:]

def dice_samplewise(yTrue, yPred):
    yPredMask = (yPred >= 0.5)
    intersection = np.sum(yTrue * yPredMask, axis=(1, 2, 3))
    total = np.sum(yTrue, axis=(1, 2, 3)) + np.sum(yPredMask, axis=(1, 2, 3))
    return 2 * intersection / total

##### Fixed effects model #####
test_data = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True) 
model = unet.unet(pretrained_weights=UNETWEIGHTS, input_size=tupImgShape)
arrPredFE = model.predict(test_data.flow(arrTestData, arrTestLabel, shuffle=False))
arrDiceFE = dice_samplewise(arrTestLabel, arrPredFE)
del model

##### Mixed effects model #####
test_data_me = GroupedDataGenerator(arrTestData, arrTestLabel, arrTestGroups,
                                    label_type='mask',
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    shuffle=False)
test_data_me.set_dummy_encoder((nTrainGroups,))                                     
model = unet.unet_mixedeffects((nTrainGroups,), pretrained_weights=MEWEIGHTS, input_size=tupImgShape, random_int=True)
arrPredME = model.predict(test_data_me)
arrDiceME = dice_samplewise(arrTestLabel, arrPredME)
del model

t, p = scipy.stats.ttest_rel(arrDiceME, arrDiceFE)
print(f't: {t:.03f}, p: {p}')

plt.hist(arrDiceFE, density=True)
plt.xlabel('Dice coefficient')
plt.ylabel('Density')
plt.show()

plt.hist(arrDiceFE, density=True, label='Fixed effects', alpha=0.5)
plt.hist(arrDiceME, density=True, label='Random intercept', alpha=0.5)
plt.xlabel('Dice coefficient')
plt.ylabel('Density')
plt.legend()
plt.show()