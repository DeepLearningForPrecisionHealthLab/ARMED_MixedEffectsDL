'''
Extract axial slices containing tumor from the original 3D MRI volumes. 

output directory structure looks like
/train
    /image
        subjectid_site_slicenumber.png
    /mask
        subjectid_seg_slicenumber.png
/val
    ...
/test
    ...
'''
DATADIR = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate'
MINVOX = 1000 # minimum number of tumor voxels per extracted slice
OUTDIR = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices'
SEED = 7640

import os
import sys
import glob
import re
import numpy as np
sys.path.append('../')
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import nibabel
import tqdm
import numpy as np
from PIL import Image

def save_slices(strSubject, arrImg, arrSlices, strSite, strType, strPartition):
    for i in arrSlices:
        strSubjectID = strSubject.split('_')[-1]
        strSavePath = os.path.join(OUTDIR, strPartition, strType, f'{strSubjectID}_{strSite}_slice{i:03d}.png')
        os.makedirs(os.path.dirname(strSavePath), exist_ok=True)
        arrSingle = arrImg[:, :, i].astype(np.float)
        arrSingle -= arrSingle.min()
        arrSingle *= (255.0 / arrSingle.max())
        img = Image.fromarray(arrSingle.astype(np.uint8), mode='L')
        img.save(strSavePath)

# Use HK as held-out site
lsImgs = glob.glob(os.path.join(DATADIR, '[!HK]*', 'Case*[!_segmentation].nii.gz'))
lsImgs.sort()
arrImgs = np.array(lsImgs)
lsSite = [re.search(r'/(\w+)/Case', s)[1] for s in lsImgs]
arrSite = np.array(lsSite)

splitTest = StratifiedShuffleSplit(test_size=0.2, random_state=SEED)
arrTrainValIdx, arrTestIdx = next(splitTest.split(arrImgs, arrSite))
splitVal = StratifiedShuffleSplit(test_size=0.125, random_state=SEED)
arrImgsTrainVal = arrImgs[arrTrainValIdx]
arrSiteTrainVal = arrSite[arrTrainValIdx]
arrTrainIdx, arrValIdx = next(splitVal.split(arrImgsTrainVal, arrSiteTrainVal))
print('Train size:', arrTrainIdx.shape[0])
print('Val size:', arrValIdx.shape[0])
print('Test size:', arrTestIdx.shape[0])
arrImgsTrain = arrImgsTrainVal[arrTrainIdx]
arrImgsVal = arrImgsTrainVal[arrValIdx]
arrImgsTest = arrImgs[arrTestIdx]

lsImgsHK = glob.glob(os.path.join(DATADIR, 'HK', 'Case*[!_segmentation].nii.gz'))
lsImgsHK.sort()
arrImgsHK = np.array(lsImgsHK)

for strImgPath in tqdm.tqdm(np.concatenate([arrImgs, arrImgsHK]), total=arrImgs.shape[0]):   
    strSegPath = strImgPath.replace('.nii.gz', '_segmentation.nii.gz')
    if not os.path.exists(strSegPath):
        strSegPath = strImgPath.replace('.nii.gz', '_Segmentation.nii.gz')
        if not os.path.exists(strSegPath):
            raise FileNotFoundError(strSegPath)

    img = nibabel.load(strImgPath)
    imgSeg = nibabel.load(strSegPath)
    arrSeg = imgSeg.get_data()
    # Find the number of prostate voxels in each axial slice
    arrVoxelCount = arrSeg.sum(axis=(0, 1))
    # Slices with at least MINVOX voxels
    arrSliceIdx = np.where(arrVoxelCount >= MINVOX)[0]

    if np.size(arrSliceIdx) == 0:
        continue

    reg = re.search(r'/(\w*)/([\w|\d]*).nii.gz', strImgPath)
    strSite = reg[1]
    strSubject = reg[2]

    if strImgPath in arrImgsTest:
        strPartition = 'test'
    elif strImgPath in arrImgsVal:
        strPartition = 'val'
    elif strImgPath in arrImgsTrain:
        strPartition = 'train'
    else:
        strPartition = 'hk'

    save_slices(strSubject, img.get_data(), arrSliceIdx, strSite, 'image', strPartition)
    save_slices(strSubject, arrSeg, arrSliceIdx, strSite, 'mask', strPartition)
