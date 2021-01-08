import sys
import numpy as np
import cv2
import albumentations
sys.path.append('../../../external/ImageDataAugmentor')
from ImageDataAugmentor import image_data_augmentor # pylint: disable=import-error
import matplotlib.pyplot as plt

DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020'
FOLD = 0
SEED = 3

strDataPath = os.path.join(DATADIR, f'fold{FOLD}.npz')
dictData = np.load(strDataPath)
arrTrainData = np.concatenate((dictData['t1_train'], dictData['t1ce_train'], dictData['t2_train'], dictData['flair_train']), axis=-1)
arrTrainLabel = dictData['mask_train']
del dictData

def overlay(arrMask):
    arrOverlay = np.zeros(arrMask.shape + (4,))
    arrOverlay[..., 0] = arrMask
    arrOverlay[..., -1] = arrMask
    return arrOverlay

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

nSamples = arrTrainData.shape[0]   

def scale(x):
    x -= x.min()
    x /= x.max()
    return x

img_gen = image_data_augmentor.ImageDataAugmentor(augment=augmentations, 
                             input_augment_mode='image',
                             preprocess_input=scale,
                             seed=SEED)
mask_gen = image_data_augmentor.ImageDataAugmentor(augment=augmentations, 
                             input_augment_mode='mask', 
                             seed=SEED)                          

img_data = img_gen.flow(arrTrainData.astype(np.float32), shuffle=True)
mask_data = mask_gen.flow(arrTrainLabel.astype(np.float32), shuffle=True)
arrBatchData, arrBatchLabel = next(zip(img_data, mask_data))

fig, ax = plt.subplots(6, 1, dpi=300)
for i in range(6):
    arrImage = arrBatchData[i,]
    arrLabel = arrBatchLabel[i,]
    ax[i].imshow(arrImage[:, :, 1], cmap='Greys_r')
    ax[i].imshow(overlay(arrLabel.squeeze()), alpha=0.5)
    ax[i].axis('off')
plt.show()