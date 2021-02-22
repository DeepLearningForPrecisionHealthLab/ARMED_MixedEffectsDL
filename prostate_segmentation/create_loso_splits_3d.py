'''
Create leave-one-site-out splits of the multisite prostate segmentation dataset.
Splits are stratified by Site.

With each site held out once, split the remaining data into 20% test, 10% validation, and 70% train.
Save the lists of partitioned images into a pickled dict.
'''

DATA = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d_128'

import os
import re
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

lsImgs = glob.glob(os.path.join(DATA, 'image', '*.npy'))
lsImgs.sort()

df = pd.DataFrame(columns=['image', 'mask', 'subject', 'group'])
i = 0
for strImgPath in lsImgs:
    strImgFile = os.path.basename(strImgPath)
    regmatch = re.search(r'(.+)_(.*).npy', strImgFile)
    strSubject = regmatch[1]
    strGroup = regmatch[2]
    ls = glob.glob(os.path.join(DATA, 'mask', f'{strSubject}_{strGroup}.npy'))
    if len(ls) == 1:
        strMaskPath = ls[0]
    else:
        raise FileNotFoundError(os.path.join(DATA, '*', 'mask', f'{strSubject}_{strGroup}.npy'))
    df.loc[i] = [strImgPath, strMaskPath, strSubject, strGroup]
    i += 1

dictSplits = {}
# Hold the QIN-Prostate dataset out as the final external validation dataset
dfExternal = df.loc[df['group'] == 'QIN-Prostate']
df = df.loc[df['group'] != 'QIN-Prostate']

for strSite in df['group'].unique():
    dfHoldOutSite = df.loc[df['group'] == strSite]
    dfRemaining = df.loc[df['group'] != strSite]

    splitter = StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=37)
    arrTrainValIdx, arrTestIdx = next(splitter.split(dfRemaining, dfRemaining['group']))
    dfTrainVal = dfRemaining.iloc[arrTrainValIdx]
    dfTest = dfRemaining.iloc[arrTestIdx]

    splitterVal = StratifiedShuffleSplit(test_size=0.125, n_splits=1, random_state=37)
    arrTrainIdx, arrValIdx = next(splitterVal.split(dfTrainVal, dfTrainVal['group']))
    dfTrain = dfTrainVal.iloc[arrTrainIdx]
    dfVal = dfTrainVal.iloc[arrValIdx]

    dictSplits[strSite] = {'train': dfTrain, 'val': dfVal, 'test': dfTest, 'held-out site': dfHoldOutSite}

with open(os.path.join(DATA, 'LOSO_splits.pkl'), 'wb') as f:
    pickle.dump(dictSplits, f)

# Split into single train/val split with all sites evenly stratified. This will be used to 
# train the final model for validation with the external dataset
splitterFinal = StratifiedShuffleSplit(test_size=0.1, n_splits=1, random_state=37)
arrTrainIdx, arrValIdx = next(splitterFinal.split(df, df['group']))
dfTrain = df.iloc[arrTrainIdx]
dfVal = df.iloc[arrValIdx]
dictFinalSplits = {'train': dfTrain, 'val': dfVal, 'external': dfExternal}

with open(os.path.join(DATA, 'QIN_external.pkl'), 'wb') as f:
    pickle.dump(dictFinalSplits, f)