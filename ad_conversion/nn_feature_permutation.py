'''
Generate feature importance figure. Should be run after completing the HPO and
generating the final trained models. 

Usage:
python nn_feature_permutation.py --output_dir </path/to/model/directory>

See python nn_feature_permutation.py --help 
for other arguments, especially for ME and other non-conventional models.
'''
import os
import glob
import pickle
import json
import tqdm
import argparse
import pandas as pd
import numpy as np
import scipy.stats
import tensorflow as tf
from medl.crossvalidation.splitting import NestedKFoldUtil
from medl.settings import RESULTSDIR
from medl.tfutils import set_gpu
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Output directory. Specifiy absolute path or'\
                    ' relative path under RESULTSDIR')
parser.add_argument('--folds', type=str, default='./10x10_kfolds_sitecluster.pkl',
                    help='Saved nested K-folds')
parser.add_argument('--permutations', type=int, default=1000, help='Number of permutations per feature')
parser.add_argument('--site_input', action='store_true', help='Model takes additional site membership input')
parser.add_argument('--zero_re', action='store_true', help='Pass in an all-zero site membership'\
                    ' design matrix to ignore learned random effects.')
args = parser.parse_args()

strFoldsPath = args.folds
strOutputDir = args.output_dir
if not strOutputDir.startswith(os.path.sep):
    strOutputDir = os.path.join(RESULTSDIR, strOutputDir)

# Load nested CV splits
with open(strFoldsPath, 'rb') as f:
    kfolds: NestedKFoldUtil = pickle.load(f) 

# Load pretty feature names
with open('../data/adni/baseline_features_24mo_feature_names.json', 'r') as f:
    dictFeatureNames = json.load(f)

lsFoldDirs = glob.glob(os.path.join(strOutputDir, 'fold*'))
lsFoldDirs.sort()

lsFolds = []

# Loop over outer folds
for strFoldDir in lsFoldDirs:
    iFold = int(strFoldDir.split('fold')[-1])
    strFoldOutputDir = os.path.join(strFoldDir, 'best_model')
    
    model = tf.keras.models.load_model(strFoldOutputDir)
    
    dfXTrain, dfZTrain, dfYTrain, _, _, _ = kfolds.get_fold(iFold)
    
    scaler = StandardScaler()
    imputer = SimpleImputer()
    
    arrXTrain = scaler.fit_transform(dfXTrain)
    arrXTrain = imputer.fit_transform(arrXTrain)

    dfPVal = pd.DataFrame(columns=['p'], 
                          index=[dictFeatureNames[x] for x in dfXTrain.columns], 
                          dtype=float)

    nPerms = args.permutations
    
    if args.site_input:
        if args.zero_re:
            # Use all-zero design matrix to isolate fixed effects
            arrZ = np.zeros_like(dfZTrain.values)
        else:
            # Use original design matrix
            arrZ = dfZTrain.values
    
        arrYPred = model.predict((arrXTrain, arrZ))
        
    else:
        arrYPred = model.predict(arrXTrain)
        
    # Baseline AUROC
    aurocBase = roc_auc_score(dfYTrain, arrYPred)
    
    # Loop over features
    for iFeature in tqdm.tqdm(range(dfXTrain.shape[1]), total=dfXTrain.shape[1]):
        strFeature = dfXTrain.columns[iFeature]
        strFeature = dictFeatureNames[strFeature]
        
        arrXPerm = arrXTrain.copy()
        arrPermIdx = np.arange(arrXTrain.shape[0])
        
        nOverBase = 0
        
        for iPerm in range(nPerms):
            np.random.seed(iPerm)
            np.random.shuffle(arrPermIdx)
            arrXPerm[:, iFeature] = arrXTrain[arrPermIdx, iFeature]

            if args.site_input:
                arrYPred = model.predict((arrXPerm, arrZ))
            else:
                arrYPred = model.predict(arrXPerm)
                
            auroc = roc_auc_score(dfYTrain, arrYPred)
            
            if aurocBase < auroc:
                nOverBase += 1
                
        dfPVal.loc[strFeature, 'p'] = nOverBase / nPerms

    dfPVal['Fold'] = iFold
    lsFolds += [dfPVal]
    
dfPValAll = pd.concat(lsFolds)
dfPValAll['Feature'] = dfPValAll.index
dfPValAll.reset_index(inplace=True, drop=True)

dfPValAll.to_csv(os.path.join(strOutputDir, 'feature_permutation.csv'))