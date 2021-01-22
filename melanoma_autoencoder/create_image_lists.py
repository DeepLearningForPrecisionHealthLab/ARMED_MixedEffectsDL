'''
Get paths to melanoma cell .png images, split into train/val/test, and parse metadata from the file paths. 
------
Kevin Nguyen
1/18/2021
'''
OUTDIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma/'
PDX = ['m481', 'm634', 'm498', 'm610'] # PDXs to include in main training dataset
DATES = ['160802', '160808', '161209', '161214', '161220', '161224'] # Dates of acquisition in common across these PDXs
MELANOMA = ['a375'] # melanoma cell lines to include in "external" validation dataset
# a375 is a BRAF V600E, NRAS wild-type cell line predicted in the paper as most aggressive.
# mv3 is a BRAF wild-type, NRAS-mutated cell line predicted in the paper as least aggressive.
MELANOCYTE = ['m116'] # untransformed melanocyte cell lines to include in external validation dataset
IMAGES_PER_CELL = 50 # images to sample from each cell's timeseries

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
sys.path.append('../')
from medl.grouped_cv import StratifiedGroupShuffleSplit #pylint: disable=import-error

lsImages = []
for strPDX in PDX:
    for strDate in DATES:
        lsCells = glob.glob(f'/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma_hackathon/raw/png/*/{strPDX}/{strDate}*/*')
        # Random select a subset of images in each cell directory
        for strCellDir in lsCells:
            lsCellImages = glob.glob(os.path.join(strCellDir, '*.png'))
            lsImages += list(np.random.choice(lsCellImages, size=IMAGES_PER_CELL))

print(len(lsImages), 'PDX images selected')
df = pd.DataFrame({'image': lsImages}, index=range(len(lsImages)))
# Paths have the format png/met_efficiency/cell_type/experiment_date/...
df['date'] = df['image'].apply(lambda x: re.search(r'(\d+)_', os.path.basename(x))[1])
df['celltype'] = df['image'].apply(lambda x: re.search(r'\d+_([a-z0-9]+)', os.path.basename(x))[1])
df['met-eff'] = df['image'].apply(lambda x: re.search(r'(high|low|na)', x)[1])
df['cell'] = df['image'].apply(lambda x: x.split(os.path.sep)[-2])

# Stratify splits so that each partition has similar representation of cell type and acquisition dates
dfStrat = df.apply(lambda row: '_'.join([row['celltype'], row['date']]), axis=1)

# Split into 70% train, 10% validation, 20% test
splitTest = StratifiedGroupShuffleSplit(test_size=0.2, n_splits=1, random_state=38)
arrIdxTrainVal, arrIdxTest = next(splitTest.split(df, 
                                                  dfStrat.values.astype(str), 
                                                  groups=df['cell'].values.astype(str)))
dfTrainVal = df.iloc[arrIdxTrainVal]
dfTest = df.iloc[arrIdxTest]
splitVal = StratifiedGroupShuffleSplit(test_size=0.125, n_splits=1, random_state=38) # 0.125 x 0.8 = 0.1
arrIdxTrain, arrIdxVal = next(splitVal.split(dfTrainVal, 
                                             dfStrat.iloc[arrIdxTrainVal].values.astype(str), 
                                             groups=dfTrainVal['cell'].values.astype(str)))
dfTrain = dfTrainVal.iloc[arrIdxTrain]
dfVal = dfTrainVal.iloc[arrIdxVal]

dfTrain.to_csv(os.path.join(OUTDIR, 'data_train.csv'))
dfVal.to_csv(os.path.join(OUTDIR, 'data_val.csv'))
dfTest.to_csv(os.path.join(OUTDIR, 'data_test.csv'))

print(dfTrain.shape[0], 'train images')
print(dfVal.shape[0], 'val images')
print(dfTest.shape[0], 'test images')

lsImagesExt = []
for strLine in MELANOMA + MELANOCYTE:
    lsCells = glob.glob(f'/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma_hackathon/raw/png/na/{strLine}/*/*')
    # Random select a subset of images in each cell directory
    for strCellDir in lsCells:
        lsCellImages = glob.glob(os.path.join(strCellDir, '*.png'))
        lsImagesExt += list(np.random.choice(lsCellImages, size=IMAGES_PER_CELL))

print(len(lsImagesExt), 'melanoma/melanocyte cell line images selected')
dfExt = pd.DataFrame({'image': lsImagesExt}, index=range(len(lsImagesExt)))
# Paths have the format png/met_efficiency/cell_type/experiment_date/...
dfExt['date'] = dfExt['image'].apply(lambda x: re.search(r'(\d+)_', os.path.basename(x))[1])
dfExt['celltype'] = dfExt['image'].apply(lambda x: re.search(r'\d+_([a-z0-9]+)', os.path.basename(x))[1])
dfExt['met-eff'] = dfExt['image'].apply(lambda x: re.search(r'(high|low|na)', x)[1])
dfExt['cell'] = dfExt['image'].apply(lambda x: x.split(os.path.sep)[-2])
dfExt.to_csv(os.path.join(OUTDIR, 'data_ext.csv'))