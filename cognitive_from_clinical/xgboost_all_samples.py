'''
As a preliminary experiment, train an XGBoost classifier to predict 24-month MCI
conversion to dementia. Use a demographic, clinical, cognitive, and imaging features
compiled in the ADNIMERGE table. 

ADNI1 used for training, and ADNI2 used for external validation
-----
Kevin Nguyen
12/18/2020
'''
DATAPATH = '../data/ADNIMERGE.csv'
DATAPATH_ADNIALL = 'adnimerge_mci.csv'
LABELPATH_ADNIALL = 'conversion_by_month.csv'
FEATURESPATH = 'candidate_current_features.csv'
MONTHS = '24'
SEED = 858
TESTSPLIT = 0.2
MODELS = 100


import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, \
    roc_curve, precision_recall_curve, average_precision_score
from scipy.stats import distributions
import matplotlib.pyplot as plt
sys.path.append('../')
from medl.grouped_cv import StratifiedGroupKFold # pylint: disable=import-error
from utils import convert_adnimerge_categorical

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
dfData = convert_adnimerge_categorical(dfData)

# Train/test split
# Convert subject IDs to integer labels
subjectEncoder = LabelEncoder()
arrSubjects = subjectEncoder.fit_transform(dfTarget.index.get_level_values(0))
# Using the KFold class since the ShuffleSplit class requires that all samples per group have the same label
splitterOuter = StratifiedGroupKFold(n_splits=int(1/TESTSPLIT), shuffle=True, random_state=SEED)
arrIndTrain, arrIndTest = next(splitterOuter.split(dfData, dfTarget, groups=arrSubjects))
dfDataTrain = dfData.iloc[arrIndTrain]
dfTargetTrain = dfTarget.iloc[arrIndTrain]
arrSubjectsTrain = arrSubjects[arrIndTrain]
dfDataTest = dfData.iloc[arrIndTest]
dfTargetTest = dfTarget.iloc[arrIndTest]
print('Train split contains {} samples with {:.03f}% converters'.format(dfDataTrain.shape[0], dfTargetTrain.mean()* 100))
print('Test split contains {} samples with {:.03f}% converters'.format(dfDataTest.shape[0], dfTargetTest.mean()* 100))

splitterInner = StratifiedGroupKFold(n_splits=5, random_state=SEED)
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

search.fit(dfDataTrain, dfTargetTrain, groups=arrSubjectsTrain)                            
# Print train performance
dfResults = pd.DataFrame(search.cv_results_)
print(dfResults[['mean_train_roc_auc', 'mean_train_balanced_accuracy']].loc[search.best_index_])

arrTestPredict = search.predict_proba(dfDataTest)[:, 1]
fAUROCTest = search.score(dfDataTest, dfTargetTest)
fAvePrec = average_precision_score(dfTargetTest.values, arrTestPredict)
print(f'Test AUROC: {fAUROCTest:.05f}')
fAcc = accuracy_score(dfTargetTest.values, arrTestPredict >= 0.5)
print(f'Bal. accuracy: {fAcc:.05f} at 0.5 threshold')
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

# Load ADNI2 data
dfDataAdni2 = pd.read_csv(DATAPATH_ADNIALL, index_col=[0, 1])
dfConversionsAdni2 = pd.read_csv(LABELPATH_ADNIALL, index_col=[0, 1])
dfConversionsAdni2 = dfConversionsAdni2.loc[dfConversionsAdni2['Study'] == 'ADNI2']
dfDataAdni2 = dfDataAdni2.loc[dfConversionsAdni2.index]
dfDataAdni2 = dfDataAdni2[lsFeatures]
dfTargetAdni2 = dfConversionsAdni2[MONTHS].loc[dfDataAdni2.index]

# Drop subjects with unknown conversion status
dfTargetAdni2 = dfTargetAdni2.loc[dfTargetAdni2 != 'unknown']
# Drop subjects who are also in ADNI1
arrOverlap = np.intersect1d(dfTarget.index.get_level_values(0),
                            dfTargetAdni2.index.get_level_values(0))
dfTargetAdni2 = dfTargetAdni2.drop(index=arrOverlap)

dfDataAdni2 = dfDataAdni2.loc[dfTargetAdni2.index]
print('Validating on', dfDataAdni2.shape[0], 'samples from ADNI2')
print('from', dfTargetAdni2.index.get_level_values(0).unique().shape[0], 'subjects')

# Convert conversion labels to int
dfTargetAdni2 = (dfTargetAdni2 == 'True').astype(int)
print(f'{100*dfTargetAdni2.mean():.02f}% of the samples are converters')

# Convert categorical variables to one-hot
dfDataAdni2 = convert_adnimerge_categorical(dfDataAdni2)
# ADNI2 has some races not present in ADNI1, make sure that the columns line up
dfDataAdni2 = dfDataAdni2[dfData.columns]

arrAdni2Predict = search.predict_proba(dfDataAdni2)[:, 1]
fAUROCTest = search.score(dfDataAdni2, dfTargetAdni2)
fAvePrec = average_precision_score(dfTargetAdni2.values, arrAdni2Predict)
print(f'Test AUROC: {fAUROCTest:.05f}')
fAcc = accuracy_score(dfTargetAdni2.values, arrAdni2Predict >= 0.5)
print(f'Bal. accuracy: {fAcc:.05f} at 0.5 threshold')

# ROC and Precision-recall curves
arrFPR, arrTPR, _ = roc_curve(dfTargetAdni2.values, arrAdni2Predict)
plt.plot(arrFPR, arrTPR, 'b-', label='ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'ROC curve, AUROC: {fAUROCTest:.03f}')
plt.show()

arrPrec, arrRec, _ = precision_recall_curve(dfTargetAdni2.values, arrAdni2Predict)
plt.plot(arrRec, arrPrec, 'r-', label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title(f'Precision-Recall curve, Ave. Precision: {fAvePrec:.03f}')
plt.show()