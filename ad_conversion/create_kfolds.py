DATAPATH = '../data/adni/baseline_features_24mo_imputeddx.csv'
LABELSPATH = '../data/adni/target_24mo_imputeddx.csv'

lsFeatures = ['APOE4', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 
              'PTEDUCAT', 'AGE', 'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 
              'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 
              'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'TRABSCOR_bl', 
              'FAQ_bl', 'mPACCdigit_bl', 'mPACCtrailsB_bl', 'Ventricles_bl', 
              'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 
              'MidTemp_bl', 'ICV_bl', 'FDG_bl', 'PTAU_bl', 'TAU_bl', 'ABETA_bl']

MINSUBJECTS = 5
OUTERFOLDS = 10
INNERFOLDS = 10

import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from medl.crossvalidation.splitting import NestedKFoldUtil

dfData = pd.read_csv(DATAPATH, index_col=0)
dfTarget = pd.read_csv(LABELSPATH, index_col=0)

# Select sites with more than MINSUBJECTS 
dfSiteCounts = dfData['SITE'].value_counts(ascending=False)
lsKeptSites = list(dfSiteCounts.index[dfSiteCounts >= MINSUBJECTS])
dfTarget = dfTarget.loc[dfData['SITE'].isin(lsKeptSites)]
dfData = dfData.loc[dfData['SITE'].isin(lsKeptSites)]

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

# One-hot encode the site
onehotSite = OneHotEncoder(sparse=False)
arrSite = onehotSite.fit_transform(dfSite.values.reshape((-1, 1)))
arrSiteNames = onehotSite.get_feature_names()
dfSiteOnehot = pd.DataFrame(arrSite, index=dfData.index, columns=[x.split('_')[1] for x in arrSiteNames])

kfolds = NestedKFoldUtil(dfData, dfSiteOnehot, dfTarget.values,
                         n_folds_outer=10, n_folds_inner=10)

# Make sure every site is represented in every training set
for iFold in range(OUTERFOLDS):
    _, dfZTrain, _, _, dfZTest, _ = kfolds.get_fold(iFold)
    assert dfZTrain.sum(axis=0).min() > 0
    
    for iFoldInner in range(INNERFOLDS):
        _, dfZTrain, _, _, dfZTest, _ = kfolds.get_fold(iFold)
        assert dfZTrain.sum(axis=0).min() > 0

with open('10x10_kfolds_sitecluster.pkl', 'wb') as f:
    pickle.dump(kfolds, f)