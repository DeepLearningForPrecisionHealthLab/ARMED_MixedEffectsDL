'''
Convert downloaded DICOMs from I2CVB to NIfTI format.

Use KevinPipelines environment which contains nipype
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
strDicomRootDir = '/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/I2CVB15/DICOM'
strOutDir = '/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209/I2CVB15'

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

dictCancerVox = {}

for strSubDir in glob.glob(os.path.join(strDicomRootDir, 'Patient*')):
    for strDicomDir in glob.glob(os.path.join(strSubDir, '*')):
        # Parse subject ID and contrast
        regmatch = re.search(r'Patient(\d+)/((\w|\d)*)', strDicomDir)
        strSub = regmatch[1]
        strContrast = regmatch[2]
        if strContrast == 'GT':
            lsSegDirs = glob.glob(os.path.join(strDicomDir, '*'))
            for strSegDir in lsSegDirs:
                strLabel = os.path.basename(strSegDir)
                strOutPath = f'sub-{strSub}/sub-{strSub}_desc-{strLabel}_mask'
                strConvertedPath = convert(strSegDir, strOutPath)
                # Check if this image contains any labeled cancer region
                if strLabel == 'cap':
                    img = nilearn.image.load_img(strConvertedPath)
                    dictCancerVox[strSub] = (img.get_data() > 0).sum()
        elif strContrast == 'T2W':
            strOutPath = f'sub-{strSub}/sub-{strSub}_{strContrast}'
            convert(strDicomDir, strOutPath)
        # Skip the DCE, MRSI, and DWI images for now
df = pd.Series(dictCancerVox)
print(df.loc[df > 0].shape[0], 'images with cancer')