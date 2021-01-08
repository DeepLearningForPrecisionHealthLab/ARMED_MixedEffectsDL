'''
Compute feature importance for the FFNN through permutation feature importance.
'''

SAVEPATH = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/mci_conversion/baseline_ffnn/20201229/best_model.h5'
SPLITS = 'splits_20test_5inner.pkl'
PERMUTATIONS = 100

import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

with open(SPLITS, 'rb') as f:
    dictSplits = pickle.load(f)
dfDataTrain, dfLabelsTrain = dictSplits['trainval']

preproc = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer())])                                                  
arrXTrain = preproc.fit_transform(dfDataTrain)
arrYTrain = dfLabelsTrain.values

model = tf.keras.models.load_model(SAVEPATH)
fAUROCBase = model.evaluate(arrXTrain, arrYTrain, verbose=0)[1]

arrImportance = np.zeros((PERMUTATIONS, arrXTrain.shape[1]))
for iFeature in range(arrXTrain.shape[1]):
    for iPerm in range(PERMUTATIONS):
        arrXPerm = arrXTrain.copy()
        np.random.shuffle(arrXPerm[:, iFeature])
        fAUROCNew = model.evaluate(arrXPerm, arrYTrain, verbose=0)[1]
        arrImportance[iPerm, iFeature] = fAUROCBase - fAUROCNew

dfImportanceWide = pd.DataFrame(arrImportance, columns=dfDataTrain.columns)
dfImportance = pd.melt(dfImportanceWide, var_name='Feature', value_name='Importance')
lsOrder = dfImportanceWide.mean(axis=0).sort_values(ascending=False).index.tolist()
plt.figure(figsize=(4, 8))
sns.barplot(data=dfImportance, y='Feature', x='Importance', orient='h', order=lsOrder)
plt.show()