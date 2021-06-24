DATAPATH = '../data/adni/baseline_features_24mo_imputeddx.csv'
LABELSPATH = '../data/adni/target_24mo_imputeddx.csv'

OUTPATH = 'nested_kfolds_24mo_5sites.pkl'

lsFeatures = ['APOE4', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 
              'PTEDUCAT', 'AGE', 'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 
              'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 
              'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'TRABSCOR_bl', 
              'FAQ_bl', 'mPACCdigit_bl', 'mPACCtrailsB_bl', 'Ventricles_bl', 
              'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 
              'MidTemp_bl', 'ICV_bl', 'FDG_bl', 'PTAU_bl', 'TAU_bl', 'ABETA_bl']

OUTERFOLDS = 5
INNERFOLDS = 5 
SEED = 8443

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold


dfData = pd.read_csv(DATAPATH, index_col=0)
dfTarget = pd.read_csv(LABELSPATH, index_col=0)

# Select the largest 5 sites
lsSites = [23, 27, 72, 128, 137]
dfTarget = dfTarget.loc[dfData['SITE'].isin(lsSites)]
dfData = dfData.loc[dfData['SITE'].isin(lsSites)]

# Select a subset of features which have the fewest missing values
dfSite = dfData['SITE']
dfData = dfData[lsFeatures]
dfTarget = dfTarget.loc[dfData.index]

# One-hot encode the categorical features
dfContinuous = dfData.iloc[:, 5:]
dfCategorical = dfData.iloc[:, :5]
# There are a small number of missing APOE4 values to handle
dfCategorical.loc[dfCategorical['APOE4'].isna(), 'APOE4'] = 99.0

onehot = OneHotEncoder(sparse=False)
arrOnehot = onehot.fit_transform(dfCategorical)
arrOnehotNames = onehot.get_feature_names()
# Replace xn_ prefix created by OneHotEncoder with the original feature names
dictCatFeatures = {'x'+ str(i): v for i, v in enumerate(dfCategorical.columns)}
lsOnehotNames = []
for strCatName in arrOnehotNames:
    strFeatureName = dictCatFeatures[strCatName.split('_')[0]]
    lsOnehotNames += [strFeatureName + '_' + strCatName.split('_')[1]]

dfOnehot = pd.DataFrame(arrOnehot, index=dfCategorical.index, columns=lsOnehotNames)
if 'APOE4_99.0' in dfOnehot.columns:
    dfOnehot.pop('APOE4_99.0')

dfData = pd.concat([dfOnehot, dfContinuous], axis=1)

# # Replace feature names with more readable ones
# with open('adnimerge_feature_names.json', 'r') as f:
#     dictFeatureNames = json.load(f)
# dfData.columns = [dictFeatureNames[x] if x in dictFeatureNames else dictFeatureNames[x + '_bl'] for x in dfData.columns ]

# One-hot encode the site
onehotSite = OneHotEncoder(sparse=False)
arrSite = onehotSite.fit_transform(dfSite.values.reshape((-1, 1)))
arrSiteNames = onehotSite.get_feature_names()
dfSiteOnehot = pd.DataFrame(arrSite, index=dfData.index, columns=[x.split('_')[1] for x in arrSiteNames])

# Save splits as a list of dicts
lsFolds = []

# Stratify by site and target
lsStrat = ['{}_{}'.format(x,y) for x, y in zip(dfSite.values, dfTarget['0'].values)]
arrStrat = np.array(lsStrat)

outersplit = StratifiedKFold(n_splits=OUTERFOLDS, shuffle=True, random_state=SEED)
for iOuter, (arrTrainValIdx, arrTestIdx) in enumerate(outersplit.split(dfData, arrStrat)):
    
    dfDataTrainVal = dfData.iloc[arrTrainValIdx]
    dfTargetTrainVal = dfTarget.iloc[arrTrainValIdx]
    dfSiteTrainVal = dfSiteOnehot.iloc[arrTrainValIdx]
    
    dfDataTest = dfData.iloc[arrTestIdx]
    dfTargetTest = dfTarget.iloc[arrTestIdx]
    dfSiteTest = dfSiteOnehot.iloc[arrTestIdx]
    
    # ensure train/val/test subjects are disjoint
    assert len(dfDataTrainVal.index.intersection(dfDataTest.index)) == 0
    
    arrStratTrainVal = arrStrat[arrTrainValIdx]
    
    lsInnerFolds = []
    innersplit = StratifiedKFold(n_splits=INNERFOLDS, shuffle=True, random_state=SEED)
    for iInner, (arrTrainIdx, arrValIdx) in enumerate(innersplit.split(dfDataTrainVal, arrStratTrainVal)):
        
        dfDataTrain = dfDataTrainVal.iloc[arrTrainIdx]
        dfTargetTrain = dfTargetTrainVal.iloc[arrTrainIdx]
        dfSiteTrain = dfSiteTrainVal.iloc[arrTrainIdx]
        
        dfDataVal = dfDataTrainVal.iloc[arrValIdx]
        dfTargetVal = dfTargetTrainVal.iloc[arrValIdx]
        dfSiteVal = dfSiteTrainVal.iloc[arrValIdx]
    
        # ensure train/val/test subjects are disjoint
        assert len(dfDataTrain.index.intersection(dfDataVal.index)) == 0
    
        lsInnerFolds += [(dfDataTrain, dfSiteTrain, dfTargetTrain, dfDataVal, dfSiteVal, dfTargetVal)]
    
    lsFolds += [{'outer': (dfDataTrainVal, dfSiteTrainVal, dfTargetTrainVal, dfDataTest, dfSiteTest, dfTargetTest),
                 'inner': lsInnerFolds}]
    
with open(OUTPATH, 'wb') as f:
    pickle.dump(lsFolds, f)
