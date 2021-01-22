'''
Get paths to melanoma cell .png images, split into train/val/test, and parse metadata from the file paths. 

To enhance batch effects, sample images from each day that are representative of the image attributes of 
that day. i.e. select images from day 1 whose brightness is near the mean for day 1.
------
Kevin Nguyen
1/18/2021
'''

OUTDIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma/filtered'
PDX = ['m481', 'm634', 'm498', 'm610'] # PDXs to include in main training dataset
DATES = ['160802', '160808', '161209', '161214', '161220', '161224'] # Dates of acquisition in common across these PDXs
IMAGESPERBATCH = 500 # images selected per PDX per day

dictCriteria = {'160802': {'Brightness': (15, 25),
                           'S.D.': (25, 35),
                           'VoL': (10, 35),
                           'SNR': (0.5, 1.5)},
                '160808': {'Brightness': (10, 30),
                           'S.D.': (15, 30),
                           'VoL': (8, 18),
                           'SNR': (0.5, 1.0)},
                '161209': {'Brightness': (12, 15),
                           'S.D.': (20, 30),
                           'VoL': (6, 10),
                           'SNR': (0.5, 0.65)},
                '161214': {'Brightness': (30, 50),
                           'S.D.': (40, 60),
                           'VoL': (15, 20),
                           'SNR': (0.60, 1.00)},
                '161220': {'Brightness': (17, 25),
                           'S.D.': (30, 40),
                           'VoL': (15, 30),
                           'SNR': (0.6, 0.9)},
                '161224': {'Brightness': (14, 17),
                           'S.D.': (20, 35),
                           'VoL': (9, 14),
                           'SNR': (0.4, 1.0)}}

# Figure labels for each PDX and day
dictPDX = {'m481': 'PDX 1 (High ME)',
           'm634': 'PDX 2 (High ME)',
           'm498': 'PDX 3 (Low  ME)',
           'm610': 'PDX 4 (Low  ME)'}
dictDates = {'160802': 'Day 1',
             '160808': 'Day 2',
             '161209': 'Day 3',
             '161214': 'Day 4',
             '161220': 'Day 5',
             '161224': 'Day 6'}

import os
import sys
import re
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('../')
from medl.grouped_cv import StratifiedGroupShuffleSplit #pylint: disable=import-error

dfImages = pd.read_csv('image_attributes.csv', index_col=0)
dfImages['Date'] = dfImages['Date'].astype(str)
lsData = []

fig, ax = plt.subplots(6, 12, figsize=(20, 8))

for iPDX, strPDX in enumerate(PDX):
    for iDate, strDate in enumerate(DATES):
        dfBatch = dfImages.loc[(dfImages['PDX'] == strPDX) & (dfImages['Date'] == strDate)]
        dictFilters = dictCriteria[strDate]
        dfFiltered = dfBatch.loc[(dfBatch['Brightness'] >= dictFilters['Brightness'][0])
                                & (dfBatch['Brightness'] <= dictFilters['Brightness'][1])
                                & (dfBatch['S.D.'] >= dictFilters['S.D.'][0])
                                & (dfBatch['S.D.'] <= dictFilters['S.D.'][1])
                                & (dfBatch['VoL'] >= dictFilters['VoL'][0])
                                & (dfBatch['VoL'] <= dictFilters['VoL'][1])
                                & (dfBatch['SNR'] >= dictFilters['SNR'][0])
                                & (dfBatch['SNR'] <= dictFilters['SNR'][1])]
        if IMAGESPERBATCH <= dfFiltered.shape[0]:
            dfSample = dfFiltered.sample(IMAGESPERBATCH)   
        else:
            dfSample = dfFiltered.sample(dfFiltered.shape[0])
        # Rename columns for compatibility with other code
        dfData = dfSample[['Date', 'PDX', 'Cell', 'Path']]
        dfData.rename(columns={'Date': 'date', 'PDX': 'celltype', 'Cell': 'cell', 'Path': 'image'}, inplace=True)
        dfData['met-eff'] = dfData['image'].apply(lambda x: re.search(r'(high|low|na)', x)[1])
        lsData += [dfData]

        for i in range(3):
            strImgPath = dfData['image'].iloc[i]
            img = Image.open(strImgPath)
            ax[iDate, iPDX * 3 + i].imshow(np.array(img), cmap='gray')
            ax[iDate, iPDX * 3 + i].axis('off')
        
    ax[0, iPDX * 3 + 1].set_title(dictPDX[strPDX])
for iRow in range(6):
    ax[iRow, 0].set_ylabel(dictDates[DATES[iRow]])
fig.tight_layout()
plt.show()

dfSelection = pd.concat(lsData, axis=0)        
print(dfSelection['celltype'].value_counts())
print(dfSelection['date'].value_counts())

os.makedirs(OUTDIR, exist_ok=True)
# Stratify splits so that each partition has similar representation of cell type and acquisition dates
dfStrat = dfSelection.apply(lambda row: '_'.join([row['celltype'], row['date']]), axis=1)

# Split into 70% train, 10% validation, 20% test
splitTest = StratifiedGroupShuffleSplit(test_size=0.2, n_splits=1, random_state=38)
arrIdxTrainVal, arrIdxTest = next(splitTest.split(dfSelection, 
                                                  dfStrat.values.astype(str), 
                                                  groups=dfSelection['cell'].values.astype(str)))
dfTrainVal = dfSelection.iloc[arrIdxTrainVal]
dfTest = dfSelection.iloc[arrIdxTest]
splitVal = StratifiedGroupShuffleSplit(test_size=0.125, n_splits=1, random_state=38) # 0.125 x 0.8 = 0.1
arrIdxTrain, arrIdxVal = next(splitVal.split(dfTrainVal, 
                                             dfStrat.iloc[arrIdxTrainVal].values.astype(str), 
                                             groups=dfTrainVal['cell'].values.astype(str)))
dfTrain = dfTrainVal.iloc[arrIdxTrain]
dfVal = dfTrainVal.iloc[arrIdxVal]

dfTrain.to_csv(os.path.join(OUTDIR, 'data_train.csv'))
dfVal.to_csv(os.path.join(OUTDIR, 'data_val.csv'))
dfTest.to_csv(os.path.join(OUTDIR, 'data_test.csv'))

print(dfTrain.shape[0], 'train images')
print(dfVal.shape[0], 'val images')
print(dfTest.shape[0], 'test images')