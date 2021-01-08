DATAPATH = '../adnimerge_mci_adni1.csv'
LABELPATH = '../conversion_by_month_adni1.csv'
DATAPATH_ADNIALL = '../adnimerge_mci.csv'
LABELPATH_ADNIALL = '../conversion_by_month.csv'
FEATURESPATH = '../candidate_current_features.csv'
MONTHS = '24'

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import convert_adnimerge_categorical # pylint: disable=import-error

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

with open('adni2_validation.pkl', 'wb') as f:
    pickle.dump((dfDataAdni2, dfTargetAdni2), f)