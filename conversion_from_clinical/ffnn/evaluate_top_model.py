'''
After running BOHB with Tune, evaluate the top performing model 
on the test partition and on ADNI2 (external dataset)
'''
SEED = 3797
SPLITS = 'splits_20test_5inner.pkl'
EPOCHS = 100
BATCHSIZE = 32
ADNI2 = 'adni2_validation.pkl'
# Path to BOHB output directory
OUTDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/mci_conversion/baseline_ffnn/20201229' 
# where to save trained model
SAVEPATH = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/mci_conversion/baseline_ffnn/20201229/best_model.h5'

import sys
import pickle
from ray.tune.analysis import Analysis
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.append('../../')
from medl import tune_models # pylint: disable=import-error

results = Analysis(OUTDIR)
dfResults = results.dataframe(metric='val_auroc', mode='max')
fBestAUROC = dfResults['val_auroc'].max()
dictBestConfig = results.get_best_config(metric='val_auroc', mode='max')

print(f'Evaluating best model with AUROC: {fBestAUROC:.03f}')
print(dictBestConfig)

with open(SPLITS, 'rb') as f:
    dictSplits = pickle.load(f)
dfDataTrain, dfLabelsTrain = dictSplits['trainval']
dfDataTest, dfLabelsTest = dictSplits['test']

# Normalize the data
preproc = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer())])                                                  
arrXTrain = preproc.fit_transform(dfDataTrain)
arrXTest = preproc.transform(dfDataTest)
arrYTrain = dfLabelsTrain.values
arrYTest = dfLabelsTest.values

model = tune_models.create_classifier_from_config(dictBestConfig, 
                                                  (arrXTrain.shape[1],), 
                                                  nMinNeurons=4)

lsCallbacks = [tf.keras.callbacks.EarlyStopping('val_auroc', mode='max', patience=5, restore_best_weights=True)]
model.fit(arrXTrain, arrYTrain,
        validation_data=(arrXTest, arrYTest),
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        callbacks=lsCallbacks,
        verbose=1)
model.save(SAVEPATH)

fAUROCTrain = model.evaluate(arrXTrain, arrYTrain, verbose=0)[1]
print(f'Train AUROC: {fAUROCTrain:.03f}')
fAUROCTest = model.evaluate(arrXTest, arrYTest, verbose=0)[1]
print(f'Test AUROC: {fAUROCTest:.03f}')
arrPredictTest = model.predict(arrXTest)

fAvePrecTest = average_precision_score(arrYTest, arrPredictTest)
# ROC and Precision-recall curves
arrFPR, arrTPR, _ = roc_curve(arrYTest, arrPredictTest)
plt.plot(arrFPR, arrTPR, 'b-', label='ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'ROC curve, AUROC: {fAUROCTest:.03f}')
plt.show()

arrPrec, arrRec, _ = precision_recall_curve(arrYTest, arrPredictTest)
plt.plot(arrRec, arrPrec, 'r-', label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title(f'Precision-Recall curve, Ave. Precision: {fAvePrecTest:.03f}')
plt.show()

# Load ADNI2 data
with open(ADNI2, 'rb') as f:
    dfDataAdni2, dfLabelsAdni2 = pickle.load(f)
arrXAdni2 = preproc.transform(dfDataAdni2)
arrYAdni2 = dfLabelsAdni2.values

fAUROCAdni2 = model.evaluate(arrXAdni2, arrYAdni2, verbose=0)[1]
print(f'ADNI2 AUROC: {fAUROCAdni2:.03f}')
arrPredictAdni2 = model.predict(arrXAdni2)

fAvePrecAdni2 = average_precision_score(arrYAdni2, arrPredictAdni2)
# ROC and Precision-recall curves
arrFPR, arrTPR, _ = roc_curve(arrYAdni2, arrPredictAdni2)
plt.plot(arrFPR, arrTPR, 'b-', label='ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(f'ROC curve, AUROC: {fAUROCAdni2:.03f}')
plt.show()

arrPrec, arrRec, _ = precision_recall_curve(arrYAdni2, arrPredictAdni2)
plt.plot(arrRec, arrPrec, 'r-', label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title(f'Precision-Recall curve, Ave. Precision: {fAvePrecAdni2:.03f}')
plt.show()