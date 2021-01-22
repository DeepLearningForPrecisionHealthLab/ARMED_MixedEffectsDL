'''
Create leave-one-site-out splits of the multisite prostate segmentation dataset.

With each site held out once, split the remaining data into 20% test and 80% train. S
Save the lists of partitioned images into a pickled dict.
'''

DATA = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices'

import os
import re
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit

lsImgs = glob.glob(os.path.join(DATA, '*', 'image', '*.png'))
lsImgs.sort()

df = pd.DataFrame(columns=['image', 'mask', 'subject', 'group'])
i = 0
for strImgPath in lsImgs:
    strImgFile = os.path.basename(strImgPath)
    regmatch = re.search(r'(.+)_(.*)_slice(\d+).png', strImgFile)
    strSubject = regmatch[1]
    strGroup = regmatch[2]
    strSlice = regmatch[3]
    ls = glob.glob(os.path.join(DATA, '*', 'mask', f'{strSubject}_{strGroup}_slice{strSlice}.png'))
    if len(ls) == 1:
        strMaskPath = ls[0]
    else:
        raise FileNotFoundError(os.path.join(DATA, '*', 'mask', f'{strSubject}_{strGroup}_slice{strSlice}.png'))
    df.loc[i] = [strImgPath, strMaskPath, strSubject, strGroup]
    i += 1

dictSplits = {}

for strSite in df['group'].unique():
    dfHoldOutSite = df.loc[df['group'] == strSite]
    dfRemaining = df.loc[df['group'] != strSite]
    lsSubjectSite = ['_'.join(x) for _, x in dfRemaining[['subject', 'group']].iterrows()]
    arrSubjectSite = np.array(lsSubjectSite).reshape((-1, 1))

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=37)
    arrTrainValIdx, arrTestIdx = next(splitter.split(dfRemaining, dfRemaining, groups=arrSubjectSite))
    dfTrainVal = dfRemaining.iloc[arrTrainValIdx]
    dfTest = dfRemaining.iloc[arrTestIdx]
    arrSubjectSiteTrainVal = arrSubjectSite[arrTrainValIdx]
    splitterVal = GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=37)
    arrTrainIdx, arrValIdx = next(splitterVal.split(dfTrainVal, dfTrainVal, groups=arrSubjectSiteTrainVal))
    dfTrain = dfTrainVal.iloc[arrTrainIdx]
    dfVal = dfTrainVal.iloc[arrValIdx]

    dictSplits[strSite] = {'train': dfTrain, 'val': dfVal, 'test': dfTest, 'held-out site': dfHoldOutSite}

with open(os.path.join(DATA, 'LOSO_splits.pkl'), 'wb') as f:
    pickle.dump(dictSplits, f)