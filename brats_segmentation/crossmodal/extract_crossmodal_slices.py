'''
Extract axial slices containing tumor from the original 3D volumes from BraTS. Each slice from each MRI contrast, plus the segmentation mask, is saved as a separate .png. 

output directory structure looks like
/train
    /image
        subjectid_contrast_slicenumber.png
    /mask
        subjectid_seg_slicenumber.png
/val
    ...
/test
    ...
'''
DATADIR = '/endosome/archive/bioinformatics/DLLab/STUDIES/BraTS2020_20201231/MICCAI_BraTS2020_TrainingData'
MINVOX = 500 # minimum number of tumor voxels per extracted slice
SLICES = 10 # slices per image to extract. Slices with the most tumor voxels will be chosen
OUTDIR = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020_separatecontrasts'
SEED = 939

import os
import sys
import glob
import pandas as pd
sys.path.append('../')
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import nibabel
import tqdm
import numpy as np
from PIL import Image

dfNames = pd.read_csv(os.path.join(DATADIR, 'name_mapping.csv'), index_col=5)
dfSurvival = pd.read_csv(os.path.join(DATADIR, 'survival_info.csv'), index_col=0)
dfMetadata = dfNames.join(dfSurvival)

def save_slices(strSubject, strContrast, arrSlices, strPartition):
    strImgPath = os.path.join(DATADIR, strSubject, strSubject + '_' + strContrast + '.nii')
    arrImg = nibabel.load(strImgPath).get_data()
    for i in arrSlices:
        strSubjectID = strSubject.split('_')[-1]
        strSavePath = os.path.join(OUTDIR, strPartition, 'image', f'{strSubjectID}_{strContrast}_slice{i:03d}.png')
        os.makedirs(os.path.dirname(strSavePath), exist_ok=True)
        arrSingle = arrImg[:, :, i].astype(np.float)
        arrSingle -= arrSingle.min()
        arrSingle *= (255.0 / arrSingle.max())
        img = Image.fromarray(arrSingle.astype(np.uint8), mode='L')
        img.save(strSavePath)

def save_mask_slices(strSubject, arrMask, arrSlices, strPartition):
    for i in arrSlices:
        strSubjectID = strSubject.split('_')[-1]
        strSavePath = os.path.join(OUTDIR, strPartition, 'mask', f'{strSubjectID}_{strContrast}_slice{i:03d}.png')
        os.makedirs(os.path.dirname(strSavePath), exist_ok=True)
        arrSingle = arrMask[:, :, i].astype(np.float)
        arrSingle -= arrSingle.min()
        arrSingle *= (255.0 / arrSingle.max())
        img = Image.fromarray(arrSingle.astype(np.uint8), mode='L')
        img.save(strSavePath)

# Partition subjects first into 80% train and 20% test, then partition train into 5 folds. 
# Stratify partitions by tumor grade and age
dfStrat = dfMetadata['Grade'].to_frame('Grade')
dfStrat['Age_Quart'] = pd.qcut(dfMetadata['Age'], q=4, labels=False)
# Assign each subject into a group labeled as tumorgrade_age
dfStrat['Stratum'] = dfStrat.apply(lambda x: '_'.join([str(xi) for xi in x]), axis=1)
# Convert to integer labels
encoderStrata = LabelEncoder()
arrStrat = encoderStrata.fit_transform(dfStrat['Stratum'])

splitTest = StratifiedShuffleSplit(test_size=0.2, random_state=SEED)
arrTrainValIdx, arrTestIdx = next(splitTest.split(dfStrat.index, arrStrat))
splitVal = StratifiedShuffleSplit(test_size=0.125, random_state=SEED)
dfStratTrainVal = dfStrat.iloc[arrTrainValIdx]
arrStratTrainVal = arrStrat[arrTrainValIdx]
arrTrainIdx, arrValIdx = next(splitVal.split(dfStratTrainVal.index, arrStratTrainVal))
print('Train size:', arrTrainIdx.shape[0])
print('Val size:', arrValIdx.shape[0])
print('Test size:', arrTestIdx.shape[0])
subjectsTrain = dfStratTrainVal.index[arrTrainIdx]
subjectsVal = dfStratTrainVal.index[arrValIdx]
subjectsTest = dfStrat.index[arrTestIdx]

for strSubject in tqdm.tqdm(dfMetadata.index, total=dfMetadata.shape[0]):
    strSegPath = os.path.join(DATADIR, strSubject, strSubject + '_seg.nii')

    imgSeg = nibabel.load(strSegPath)
    arrSeg = imgSeg.get_data()
    # Get a mask for just the tumor (label 1 is the non-enhancing/necrotic core and label 4 is the enhancing region)
    arrMask = np.zeros_like(arrSeg)
    arrMask[arrSeg == 1] = 1
    arrMask[arrSeg == 4] = 1
    # Find the number of tumor voxels in each axial slice
    arrVoxelCount = arrMask.sum(axis=(0, 1))
    # Slice indices in decreasing order of voxel count, skipping every other slice
    arrVoxelCount[::2] = 0
    arrSliceIdx = np.argsort(arrVoxelCount)
    # Slices with at least MINVOX voxels
    arrSliceIdx = arrSliceIdx[arrVoxelCount[arrSliceIdx] >= MINVOX]

    if np.size(arrSliceIdx) == 0:
        continue

    for strContrast in ['t1', 't1ce', 't2', 'flair', 'seg']:
        if strSubject in subjectsTest:
            strPartition = 'test'
        elif strSubject in subjectsVal:
            strPartition = 'val'
        else:
            strPartition = 'train'
        if strContrast == 'seg':
            save_mask_slices(strSubject, arrMask, arrSliceIdx, strPartition)
        else:
            save_slices(strSubject, strContrast, arrSliceIdx, strPartition)
