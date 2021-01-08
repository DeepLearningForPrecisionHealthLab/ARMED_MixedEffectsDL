SEED = 883
OUTDIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/medl_toy_data/diabetes'
CLUSTERS = 5
SLOPESD = 100
INTSD = 50
TESTSPLIT = 0.2
FOLDS = 5
SAVEORIG = True

import os
import sys
import pickle
import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

dfData, dfTarget = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
arrClusters = np.random.randint(0, CLUSTERS, size=(dfTarget.shape))
# Add a random slope for s5 for each cluster. Draw from normal distribution with mean zero and specified variance
np.random.seed(SEED)
arrClusterSlopes = np.random.randn(CLUSTERS) * SLOPESD
arrRandomSlopes = np.vectorize(lambda x: arrClusterSlopes[x])(arrClusters)
dfTargetRE = dfTarget.copy()
dfTargetRE += np.log(dfData['s5'] + 1) * arrRandomSlopes

# Add a random interecept for each cluster
np.random.seed(SEED * 2)
arrClusterInts = np.random.randn(CLUSTERS) * INTSD
arrRandomInts = np.vectorize(lambda x: arrClusterInts[x])(arrClusters)
dfTargetRE += arrRandomInts

# Bin the samples based on target for stratified splitting
dfTargetBin = pd.qcut(dfTarget, 5, labels=False)

splitterOuter = StratifiedShuffleSplit(test_size=TESTSPLIT, random_state=SEED)
arrIndTrain, arrIndTest = next(splitterOuter.split(dfData, dfTargetBin))
dfDataTrain = dfData.iloc[arrIndTrain]
dfTargetTrain = dfTarget.iloc[arrIndTrain]
dfTargetRETrain = dfTargetRE.iloc[arrIndTrain]
arrClustersTrain = arrClusters[arrIndTrain]
dfDataTest = dfData.iloc[arrIndTest]
dfTargetTest = dfTarget.iloc[arrIndTest]
dfTargetRETest = dfTargetRE.iloc[arrIndTest]
arrClustersTest = arrClusters[arrIndTest]

print('Train split contains {} samples with mean A1c {:.03f}'.format(dfDataTrain.shape[0], dfTargetTrain.mean()))
print('Test split contains {} samples with mean A1c {:.03f}'.format(dfDataTest.shape[0], dfTargetTest.mean()))
dictSplits = {'test': (dfDataTest, dfTargetTest, arrClustersTest),
              'trainval': (dfDataTrain, dfTargetTrain, arrClustersTrain)}
dictSplitsRE = {'test': (dfDataTest, dfTargetRETest, arrClustersTest),
              'trainval': (dfDataTrain, dfTargetRETrain, arrClustersTrain)}

splitterInner = StratifiedKFold(n_splits=FOLDS, random_state=SEED)
dfTargetBinTrain = dfTargetBin.iloc[arrIndTrain]
for iFold, (arrIndTrain, arrIndVal) in enumerate(splitterInner.split(dfDataTrain, dfTargetBinTrain)):
    dfDataInnerTrain = dfDataTrain.iloc[arrIndTrain]
    dfTargetInnerTrain = dfTargetTrain.iloc[arrIndTrain]
    dfTargetREInnerTrain = dfTargetRETrain.iloc[arrIndTrain]
    arrClustersInnerTrain = arrClustersTrain[arrIndTrain]
    dfDataInnerVal = dfDataTrain.iloc[arrIndVal]
    dfTargetInnerVal = dfTargetTrain.iloc[arrIndVal]
    dfTargetREInnerVal = dfTargetRETrain.iloc[arrIndVal]
    arrClustersInnerVal = arrClustersTrain[arrIndVal]

    dictSplits['train' + str(iFold)] = (dfDataInnerTrain, dfTargetInnerTrain, arrClustersInnerTrain)
    dictSplits['val' + str(iFold)] = (dfDataInnerVal, dfTargetInnerVal, arrClustersInnerVal)
    dictSplitsRE['train' + str(iFold)] = (dfDataInnerTrain, dfTargetREInnerTrain, arrClustersInnerTrain)
    dictSplitsRE['val' + str(iFold)] = (dfDataInnerVal, dfTargetREInnerVal, arrClustersInnerVal)
    print(f'Inner fold {iFold}')
    print('Train split contains {} samples with mean A1c {:.03f}'.format(dfDataInnerTrain.shape[0], 
                                                                        dfTargetInnerTrain.mean()))
    print('Val split contains {} samples with mean A1c {:.03f}'.format(dfDataInnerVal.shape[0], 
                                                                    dfTargetInnerVal.mean()))

if SAVEORIG:
    with open(os.path.join(OUTDIR, 'splits_orig.pkl'), 'wb') as f:
        pickle.dump(dictSplits, f)

with open(os.path.join(OUTDIR, 'splits_randomeffects.pkl'), 'wb') as f:
    pickle.dump(dictSplitsRE, f)

dfRandomEffects = pd.DataFrame({'random_slopes': arrClusterSlopes, 'random_intercepts': arrClusterInts})
dfRandomEffects.to_csv(os.path.join(OUTDIR, 'random_effects.csv'))