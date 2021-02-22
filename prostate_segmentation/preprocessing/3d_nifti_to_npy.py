'''
Downsample the full 3D MRI volumes and save as .npy files
'''

IMGDIR = '/archive/bioinformatics/DLLab/STUDIES/Prostate_Multi_Datasets_20210209'
RES = 0.75 # mm isotropic
SIZE = (128, 128, 32) # 192 IS TOO BIG for 32GB GPU, even with batch size of 4
OUTDIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d_128'

import os 
import re
import glob
import json
import numpy as np
import nibabel
import scipy.ndimage
import matplotlib.pyplot as plt

os.makedirs(os.path.join(OUTDIR, 'image'), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, 'mask'), exist_ok=True)

for strSiteDir in sorted(glob.glob(os.path.join(IMGDIR, '*'))):
    strSite = os.path.basename(strSiteDir)
    if strSite == 'I2CVB15':
        # I2CVB 1.5T data was downloaded separately and the directory structure differs
        lsT2 = glob.glob(os.path.join(strSiteDir, 'nifti', '*', '*T2W.nii.gz'))
        lsT2.sort()
        lsLabel = []
        for strT2Path in lsT2:
            # Find corresponding mask file
            strLabelPath = strT2Path.replace('T2W', 'desc-prostate_mask')
            assert os.path.exists(strLabelPath)
            lsLabel += [strLabelPath]
    elif strSite == 'QIN-Prostate':
        lsT2 = glob.glob(os.path.join(strSiteDir, 'nifti', '*', '*', 'T2w.nii.gz'))
        lsT2.sort()
        lsLabel = []
        for strT2Path in lsT2:           
            # Read the segmentation metadata to find the label no. corresponding to whole prostate
            strSegMeta = os.path.join(os.path.dirname(strT2Path), 'segmentation-meta.json')
            with open(strSegMeta, 'r') as f:
                dictMeta = json.load(f)
            for dictLabel in dictMeta['segmentAttributes']:
                dictLabel = dictLabel[0]
                if dictLabel['SegmentDescription'] == 'WholeGland':
                    iLabel = dictLabel['labelID']
                    break

            strLabelPath = strT2Path.replace('T2w', f'segmentation-{iLabel}')
            assert os.path.exists(strLabelPath)
            lsLabel += [strLabelPath]
    else:
        lsT2 = glob.glob(os.path.join(strSiteDir, '*[!(segmentation)].nii.gz'))
        lsT2.sort()
        lsLabel = []
        for strT2Path in lsT2:
            # Find corresponding mask file
            strLabelPath = strT2Path.replace('.nii.gz', '_segmentation.nii.gz')
            if not os.path.exists(strLabelPath):
                strLabelPath = strT2Path.replace('.nii.gz', '_Segmentation.nii.gz')
            assert os.path.exists(strLabelPath)
            lsLabel += [strLabelPath]

    for iSub in range(len(lsT2)):
        strT2Path = lsT2[iSub]
        strLabelPath = lsLabel[iSub]
        imgT2 = nibabel.load(strT2Path)
        imgLabel = nibabel.load(strLabelPath)

        # Resample
        arrOrigZoom = np.array(imgT2.header.get_zooms())
        arrZoomFactor = arrOrigZoom / np.array([RES,] * 3) 
        arrT2Resamp = scipy.ndimage.zoom(imgT2.get_data(), arrZoomFactor)
        arrLabelResamp = scipy.ndimage.zoom(imgLabel.get_data(), arrZoomFactor, order=0)

        # Pad a couple empty slices to facilitate cropping
        arrPad = np.zeros(arrT2Resamp.shape[:2] + (1,))
        arrT2Resamp = np.concatenate([arrPad,] * 5 + [arrT2Resamp,] + [arrPad,] * 5, axis=2)
        arrLabelResamp = np.concatenate([arrPad,] * 5 + [arrLabelResamp,] + [arrPad,] * 5, axis=2)

        # Crop around the prostate
        arrCenter = np.array(scipy.ndimage.center_of_mass(arrLabelResamp))
        arrCenter = np.round(arrCenter).astype(int)
        hMin = np.max((arrCenter[0] - SIZE[0] // 2, 0))
        hMax = hMin + SIZE[0]
        if hMax > arrT2Resamp.shape[0]:
            hMax = arrT2Resamp.shape[0]
            hMin = hMax - SIZE[0]
        wMin = np.max((arrCenter[1] - SIZE[1] // 2, 0))
        wMax = wMin + SIZE[1]
        if wMax > arrT2Resamp.shape[1]:
            wMax = arrT2Resamp.shape[1]
            wMin = wMax - SIZE[1]
        dMin = np.max((arrCenter[2] - SIZE[2] // 2, 0))
        dMax = dMin + SIZE[2]
        if dMax > arrT2Resamp.shape[2]:
            dMax = arrT2Resamp.shape[2]
            dMin = dMax - SIZE[2]
        arrT2Crop = arrT2Resamp[hMin:hMax, wMin:wMax, dMin:dMax]
        arrLabelCrop = arrLabelResamp[hMin:hMax, wMin:wMax, dMin:dMax]
        arrLabelCrop = arrLabelCrop.astype(bool)
        if arrLabelCrop.shape != SIZE:
            raise ValueError

        if (strSite == 'I2CVB15'):
            # These sites are flipped across the Anterior-Posterior axis compared to the others
            arrT2Crop = np.flip(arrT2Crop, axis=1)
            arrLabelCrop = np.flip(arrLabelCrop, axis=1)
        if (strSite == 'QIN-Prostate'):
            arrT2Crop = np.flip(arrT2Crop, axis=1)

        if strSite == 'QIN-Prostate':
            strSub = re.search(r'PCAMPMRI-(\d+)', strT2Path)[1]
        else:
            strSub = re.search(r'(\d+)', os.path.basename(strT2Path))[1]
        
        strT2OutPath = os.path.join(OUTDIR, 'image', f'{strSub}_{strSite}.npy')
        np.save(strT2OutPath, arrT2Crop)
        strLabelOutPath = os.path.join(OUTDIR, 'mask', f'{strSub}_{strSite}.npy')
        np.save(strLabelOutPath, arrLabelCrop)

# Check some random images to make sure that the label mask still aligns with the image
lsImgs = glob.glob(os.path.join(OUTDIR, 'image', '*.npy'))
lsImgs.sort()
lsMasks = glob.glob(os.path.join(OUTDIR, 'mask', '*.npy'))
lsMasks.sort()
fig, ax = plt.subplots(2, 2)
for j in range(4):
    i = np.random.choice(range(len(lsImgs)))
    print(lsImgs[i])
    arrImg = np.load(lsImgs[i]).astype(float)
    arrMask = np.load(lsMasks[i]).astype(float)
    arrImg -= arrImg.min()
    arrImg /= arrImg.max()
    ax.flatten()[j].imshow(arrImg[:, :, SIZE[2] // 2], cmap='gray')
    arrOverlay = np.zeros(SIZE[:2] + (4,))
    arrOverlay[:, :, 0] = arrMask[:, :, SIZE[2] // 2] * 0.3
    arrOverlay[:, :, 3] = arrMask[:, :, SIZE[2] // 2] * 0.3
    ax.flatten()[j].imshow(arrOverlay)