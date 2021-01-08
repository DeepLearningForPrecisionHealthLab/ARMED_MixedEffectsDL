'''
Pre-generate train/test and inner KFold splits
-----
12/28/2020
Kevin Nguyen
'''

DATAPATH = '../adnimerge_mci_adni1.csv'
LABELPATH = '../conversion_by_month_adni1.csv'
FEATURESPATH = '../candidate_current_features.csv'
MONTHS = '24'
SEED = 769
TESTSPLIT = 0.2

import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
sys.path.append('../../')
from medl.grouped_cv import StratifiedGroupKFold, StratifiedGroupShuffleSplit # pylint: disable=import-error

dfData = pd.read_csv(DATAPATH, index_col=[0, 1])
dfConversions = pd.read_csv(LABELPATH, index_col=[0, 1])
# Selected, potentially predictive features
with open(FEATURESPATH, 'r') as f:
    lsFeatures = f.read().splitlines()
dfData = dfData[lsFeatures]
dfTarget = dfConversions[MONTHS].loc[dfData.index]
# Drop subjects with unknown conversion status
dfTarget = dfTarget.loc[dfTarget != 'unknown']
dfData = dfData.loc[dfTarget.index]
print(dfData.shape[0], 'samples with data')
print('from', dfTarget.index.get_level_values(0).unique().shape[0], 'subjects')

# Convert conversion labels to int
dfTarget = (dfTarget == 'True').astype(int)
print(f'{100*dfTarget.mean():.02f}% of the samples are converters')

# Convert categorical variables to one-hot
dfGender = dfData.pop('PTGENDER')
dfData['MALE'] = dfGender == 'Male'
dfEthnicity = dfData.pop('PTETHCAT')
dfData['HISP/LATINO'] = dfEthnicity == 'Hisp/Latino'
onehot = OneHotEncoder(sparse=False, drop=['White'])
dfRace = dfData.pop('PTRACCAT')
arrRace = onehot.fit_transform(dfRace.values.reshape(-1, 1))
arrRaceCats = onehot.categories_[0]
arrRaceCats = np.delete(arrRaceCats, onehot.drop_idx_[0])
for i, strRace in enumerate(arrRaceCats):
    dfData['RACE-' + strRace.upper()] = arrRace[:, i]
onehot = OneHotEncoder(sparse=False, drop=['Married'])
dfMarriage = dfData.pop('PTMARRY')
arrMarriage = onehot.fit_transform(dfMarriage.values.reshape(-1, 1))
arrMarriageCats = onehot.categories_[0]
arrMarriageCats = np.delete(arrMarriageCats, onehot.drop_idx_[0])
for i, strMarriage in enumerate(arrMarriageCats):
    dfData['MARRIAGE-' + strMarriage.upper()] = arrMarriage[:, i]


# Convert subject IDs to integer labels
subjectEncoder = LabelEncoder()
arrSubjects = subjectEncoder.fit_transform(dfTarget.index.get_level_values(0))

# Train/test split
# splitterOuter = StratifiedGroupShuffleSplit(n_splits=1, test_size=TESTSPLIT, random_state=SEED)
# Using the KFold class since the ShuffleSplit class requires that all samples per group have the same label
splitterOuter = StratifiedGroupKFold(n_splits=int(1/TESTSPLIT), shuffle=True, random_state=SEED)
arrIndTrain, arrIndTest = next(splitterOuter.split(dfData, dfTarget, groups=arrSubjects))
dfDataTrain = dfData.iloc[arrIndTrain]
dfTargetTrain = dfTarget.iloc[arrIndTrain]
arrSubjectsTrain = arrSubjects[arrIndTrain]
dfDataTest = dfData.iloc[arrIndTest]
dfTargetTest = dfTarget.iloc[arrIndTest]

print('Train split contains {} samples from {} subjects with {:.03f}% converters'.format(dfDataTrain.shape[0], 
                                                                                         dfTargetTrain.index.get_level_values(0).unique().shape[0],
                                                                                         dfTargetTrain.mean()* 100))
print('Test split contains {} samples from {} subjects with {:.03f}% converters'.format(dfDataTest.shape[0], 
                                                                                        dfTargetTest.index.get_level_values(0).unique().shape[0],
                                                                                        dfTargetTest.mean()* 100))

def check_intersect(train, test):
    # Ensures that no subjects are in both test and train partitions
    arrIntersect = np.intersect1d(train.index.get_level_values(0), test.index.get_level_values(0))
    print(arrIntersect.shape[0], 'subjects in both splits')

check_intersect(dfTargetTrain, dfTargetTest)

dictSplits = {'test': (dfDataTest, dfTargetTest),
              'trainval': (dfDataTrain, dfTargetTrain)}

splitterInner = StratifiedGroupKFold(n_splits=5, random_state=SEED)

for iFold, (arrIndTrain, arrIndVal) in enumerate(splitterInner.split(dfDataTrain, dfTargetTrain, groups=arrSubjectsTrain)):
    dfDataInnerTrain = dfDataTrain.iloc[arrIndTrain]
    dfTargetInnerTrain = dfTargetTrain.iloc[arrIndTrain]
    dfDataInnerVal = dfDataTrain.iloc[arrIndVal]
    dfTargetInnerVal = dfTargetTrain.iloc[arrIndVal]
    dictSplits['train' + str(iFold)] = (dfDataInnerTrain, dfTargetInnerTrain)
    dictSplits['val' + str(iFold)] = (dfDataInnerVal, dfTargetInnerVal)
    print(f'Inner fold {iFold}')
    print('Train split contains {} samples from {} subjects with {:.03f}% converters'.format(dfDataInnerTrain.shape[0], 
                                                                                             dfTargetInnerTrain.index.get_level_values(0).unique().shape[0],
                                                                                             dfTargetInnerTrain.mean()* 100))
    print('Val split contains {} samples from {} subjects with {:.03f}% converters'.format(dfDataInnerVal.shape[0], 
                                                                                            dfTargetInnerVal.index.get_level_values(0).unique().shape[0],
                                                                                            dfTargetInnerVal.mean()* 100))
    check_intersect(dfTargetInnerTrain, dfTargetInnerVal)                                                                                            

with open(f'splits_20test_5inner.pkl', 'wb') as f:
    pickle.dump(dictSplits, f)