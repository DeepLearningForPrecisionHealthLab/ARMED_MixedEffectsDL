'''
Convert the data into multipage .tif format for compatibility with Joris Roel's UNET-DAT implementation.
'''
import os
import glob
import numpy as np
import tifffile as tiff
from PIL import Image

strImgTrainPath = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/train/image'
strLabelTrainPath = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/train/mask'
strImgTestPath = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/test/image'
strLabelTestPath = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/test/mask'
strImgHKPath = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/hk/image'
strLabelHKPath = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/hk/mask'

lsImgTrain = sorted(glob.glob(os.path.join(strImgTrainPath, '*')))
lsImgTest = sorted(glob.glob(os.path.join(strImgTestPath, '*')))
lsImgHK = sorted(glob.glob(os.path.join(strImgHKPath, '*')))
lsLabelTrain = sorted(glob.glob(os.path.join(strLabelTrainPath, '*')))
lsLabelTest = sorted(glob.glob(os.path.join(strLabelTestPath, '*')))
lsLabelHK = sorted(glob.glob(os.path.join(strLabelHKPath, '*')))

# Split HK into train/test. We'll use the first 83 images, which come from 6 of the 12 HK subjects, for train.
lsImgHKTrain = lsImgHK[:83]
lsImgHKTest = lsImgHK[83:]
lsLabelHKTrain = lsLabelHK[:83]
lsLabelHKTest = lsLabelHK[83:]

def concatenate_images(lsImgs, strOutPath):
    ls = []
    for strLabelPath in lsImgs:
        ls += [np.expand_dims(np.array(Image.open(strLabelPath)).astype('uint8'), axis=0)]
    arrLabels = np.concatenate(ls, axis=0)
    tiff.imwrite(strOutPath, arrLabels)

concatenate_images(lsImgTrain, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/train_image.tif')
concatenate_images(lsLabelTrain, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/train_label.tif')
concatenate_images(lsImgTest, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/test_image.tif')
concatenate_images(lsLabelTest, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/test_label.tif')
concatenate_images(lsImgHKTrain, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/hk_train_image.tif')
concatenate_images(lsLabelHKTrain, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/hk_train_label.tif')
concatenate_images(lsImgHKTest, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/hk_test_image.tif')
concatenate_images(lsLabelHKTest, '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices_tif/hk_test_label.tif')