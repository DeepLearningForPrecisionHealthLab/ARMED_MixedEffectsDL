'''
Downsample the I2CVB 1.5T data to 384x384 to match the rest of the datasets.
'''

IMGDIR = '/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/I2CVB15/nifti'
OUTDIR = '/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/I2CVB15/nifti_384x384'

import re
import os
import glob
import nibabel
import nibabel.processing

lsImg = glob.glob(os.path.join(IMGDIR, '*', '*.nii.gz'))
lsImg.sort()

for strImgPath in lsImg:
    strContrast = re.search(r'_((\w|\d)+)\.', strImgPath)[1]
    img = nibabel.load(strImgPath)
    voxsize = abs(img.affine[0, 0])
    voxnew = voxsize * img.shape[0] / 384
    if strContrast == 'T2W':
        imgResamp = nibabel.processing.resample_to_output(img, (voxnew, voxnew, img.affine[2, 2]))
    else:
        imgResamp = nibabel.processing.resample_to_output(img, 
                                                          (voxnew, voxnew, img.affine[2, 2]),
                                                          order=0)
    strOutPath = os.path.join(OUTDIR, strImgPath.split(os.path.sep)[-2], os.path.basename(strImgPath))
    os.makedirs(os.path.dirname(strOutPath), exist_ok=True)
    imgResamp.to_filename(strOutPath)