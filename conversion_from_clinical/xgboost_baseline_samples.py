'''
As a preliminary experiment, train an XGBoost classifier to predict 24-month MCI
conversion to dementia. Use a demographic, clinical, cognitive, and imaging features
compiled in the ADNIMERGE table. 
-----
Kevin Nguyen
12/18/2020
'''
DATAPATH = 'adnimerge_mci.csv'
LABELPATH = 'conversion_by_month.csv'
FEATURESPATH = 'candidate_baseline_features.csv'
MONTHS = '24'
SEED = 23450897
TESTSPLIT = 0.2
MODELS = 100

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, \
    roc_curve, precision_recall_curve, average_precision_score
from scipy.stats import distributions
import matplotlib.pyplot as plt

dictParams = {'xgboost__n_estimators': np.arange(8, 248),
              'xgboost__max_depth': np.arange(2, 8),
              'xgboost__learning_rate': distributions.loguniform(1e-4, 1e-1),
              'xgboost__booster': ['gbtree', 'gblinear', 'dart'],
              'xgboost__gamma': distributions.uniform(0.0, 0.5),
              'xgboost__min_child_weight': distributions.uniform(0.7, 5.0),
              'xgboost__max_delta_step': distributions.uniform(0, 0.2),
              'xgboost__subsample': distributions.uniform(0.3, 0.7),
              'xgboost__colsample_bytree': distributions.uniform(0.5, 0.5),
              'xgboost__colsample_bylevel': distributions.uniform(0.5, 0.5),
              'xgboost__colsample_bynode': distributions.uniform(0.5, 0.5),
              'xgboost__alpha': distributions.uniform(0.0, 0.1),
              'xgboost__lambda': distributions.uniform(0.8, 5.0)
              }

dfData = pd.read_csv(DATAPATH, index_col=[0, 1])
dfConversions = pd.read_csv(LABELPATH, index_col=[0, 1])
# Selected, potentially predictive features
with open(FEATURESPATH, 'r') as f:
    lsFeatures = f.read().splitlines()
dfData = dfData[lsFeatures]
# Keep only baseline visits
dfData = dfData.loc[dfData.index.get_level_values(1) == 0]
dfTarget = dfConversions[MONTHS].loc[dfData.index]
# Drop subjects with unknown conversion status
dfTarget = dfTarget.loc[dfTarget != 'unknown']
dfData = dfData.loc[dfTarget.index]
print(dfData.shape[0], 'subjects with baseline data')

# Convert conversion labels to int
dfTarget = (dfTarget == 'True').astype(int)
print(f'{100*dfTarget.mean():.02f}% of the subjects are converters')

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

# Train/test split
splitterOuter = StratifiedShuffleSplit(n_splits=1, test_size=TESTSPLIT, random_state=SEED)
arrIndTrain, arrIndTest = next(splitterOuter.split(dfData, dfTarget))
dfDataTrain = dfData.iloc[arrIndTrain]
dfTargetTrain = dfTarget.iloc[arrIndTrain]
dfDataTest = dfData.iloc[arrIndTest]
dfTargetTest = dfTarget.iloc[arrIndTest]

splitterInner = StratifiedKFold(n_splits=5, random_state=SEED)
model = Pipeline([('scaler', StandardScaler()),
                ('imputer', SimpleImputer()),
                ('xgboost', XGBClassifier())])
search = RandomizedSearchCV(model, 
                            dictParams, 
                            scoring=['balanced_accuracy', 'roc_auc'],
                            n_iter=MODELS, 
                            n_jobs=4,
                            refit='roc_auc',
                            cv=splitterInner,
                            random_state=SEED,
                            return_train_score=True,
                            verbose=1)

search.fit(dfDataTrain, dfTargetTrain)                            
arrTestPredict = search.predict_proba(dfDataTest)[:, 1]
fAUROCTest = search.score(dfDataTest, dfTargetTest)
fAvePrec = average_precision_score(dfTargetTest.values, arrTestPredict)
print(f'Test AUROC: {fAUROCTest:.05f}')
# fAcc = accuracy_score(dfTargetTest.values, arrTestPredict >= 0.5)
# print(f'Bal. accuracy: {fAcc:.05f} at 0.5 threshold')
# arrConfusion = confusion_matrix(dfTargetTest.values, arrTestPredict >= 0.5)
# nTruePos, nFalseNeg, nFalsePos, nTrueNeg = arrConfusion.ravel()
# fSens = nTruePos / (nTruePos + nFalseNeg)
# fSpec = nTrueNeg / (nTrueNeg + nFalsePos)
# print(f'Sensitivity: {fSens:.05f}, Specificity: {fSpec:.05f} at 0.5 threshold')

# ROC and Precision-recall curves
arrFPR, arrTPR, _ = roc_curve(dfTargetTest.values, arrTestPredict)
plt.plot(arrFPR, arrTPR, 'b-', label='ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'ROC curve, AUROC: {fAUROCTest:.03f}')
plt.show()

arrPrec, arrRec, _ = precision_recall_curve(dfTargetTest.values, arrTestPredict)
plt.plot(arrRec, arrPrec, 'r-', label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title(f'Precision-Recall curve, Ave. Precision: {fAvePrec:.03f}')
plt.show()
