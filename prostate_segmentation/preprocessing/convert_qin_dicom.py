'''
Convert DICOMs from the QIN dataset into NIfTI files.

Note that the way the affine is encoded in the converted image, it will 
not be read correctly by ITK-Snap. This results in the segmentation 
apparently not lining up with the image. However, this should not be a
problem with other NIfTI readers or viewers.

Use KevinPipelines environment which contains nipype.
'''

import os
import shutil
import glob
import re
from nipype.interfaces.dcm2nii import Dcm2niix
import nilearn
import pandas as pd

# Path to dcm2niix, which is in the mricrogl package
DCM2NIIX = '/project/bioinformatics/DLLab/softwares/mricrogl/mricrogl_lx'
strDicomRootDir = '/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/QIN-Prostate/dicom'
strOutDir = '/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/QIN-Prostate/nifti'

def convert(strInDir, strOutPath):
    converter = Dcm2niix(source_dir=strInDir,
                        output_dir=strOutDir,
                        verbose=False,
                        environ={'PATH': '$PATH:' + DCM2NIIX})
    converter.run()  
    strTempPath = converter.output_files[0]       
    strOutFullPath = os.path.join(strOutDir, strOutPath)
    os.makedirs(os.path.dirname(strOutFullPath), exist_ok=True)
    shutil.move(strTempPath, strOutFullPath + '.nii.gz')
    shutil.move(strTempPath.replace('.nii.gz', '.json'), strOutFullPath + '.json')
    return strOutFullPath + '.nii.gz'

for strDicomDir in glob.glob(os.path.join(strDicomRootDir, 'PCAMP*', '*', '*T2_Weighted_Axial-*')):
    # Parse subject ID and contrast
    regmatch = re.search(r'(PCAMPMRI-\d+)/(.*)/', strDicomDir)
    strSub = regmatch[1]
    strSession = regmatch[2]
    strOutPath = f'{strSub}/{strSession}/T2w'
    convert(strDicomDir, strOutPath)