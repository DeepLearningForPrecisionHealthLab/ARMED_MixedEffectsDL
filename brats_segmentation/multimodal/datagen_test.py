import os
import sys
import numpy as np
import cv2
import albumentations
import matplotlib.pyplot as plt
sys.path.append('../../')
from medl.datagenerator import GroupedDataGenerator #pylint: disable=import-error

DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020_groups'
SEED = 3

strDataPath = os.path.join(DATADIR, 'train_test.npz')
# Load partitioned data
dictData = np.load(strDataPath)
arrTrainData = np.concatenate((dictData['t1_train'], dictData['t1ce_train'], dictData['t2_train'], dictData['flair_train']), axis=-1)
arrTrainLabel = dictData['mask_train']
arrTrainGroups = dictData['subject_train'].reshape(-1, 1)
arrTestData = np.concatenate((dictData['t1_test'], dictData['t1ce_test'], dictData['t2_test'], dictData['flair_test']), axis=-1)
arrTestLabel = dictData['mask_test']
arrTestGroups = dictData['subject_test'].reshape(-1, 1)
del dictData
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

train_gen = GroupedDataGenerator(arrTrainData, arrTrainLabel, arrTrainGroups,
                                label_type='mask',
                                samplewise_center=True,
                                samplewise_std_normalization=True,
                                augmentation=augmentations,
                                seed=SEED)
test_gen = GroupedDataGenerator(arrTestData, arrTestLabel, arrTestGroups,
                                label_type='mask',
                                samplewise_center=True,
                                samplewise_std_normalization=True,
                                seed=SEED)                                

(arrBatchData, arrBatchGroups), arrBatchLabel = train_gen.__getitem__(5)

def overlay(arrMask):
    arrOverlay = np.zeros(arrMask.shape + (4,))
    arrOverlay[..., 0] = arrMask
    arrOverlay[..., -1] = arrMask
    return arrOverlay

fig, ax = plt.subplots(6, 1, dpi=300)
for i in range(6):
    arrImage = arrBatchData[i,]
    arrLabel = arrBatchLabel[i,]
    ax[i].imshow(arrImage[:, :, 1], cmap='Greys_r')
    ax[i].imshow(overlay(arrLabel.squeeze()), alpha=0.5)
    ax[i].axis('off')
plt.show()

nGroups = train_gen.arrGroups.shape[1]
print('Training data contains', nGroups, 'groups')
test_gen.set_dummy_encoder((nGroups,))
(arrBatchData, arrBatchGroups), arrBatchLabel = test_gen.__getitem__(5)
print('Generated a dummy design matrix for test data with shape', arrBatchGroups.shape)

print(np.concatenate(train_gen.group_encoder.categories_).shape[0])