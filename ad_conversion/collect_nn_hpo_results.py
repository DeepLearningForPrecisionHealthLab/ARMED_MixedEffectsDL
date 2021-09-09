'''
Collect results from each outer fold into a single dataframe. 

Usage: python collect_nn_hpo_results.py --output_dir </path/to/hpo/output/dir>

This creates a combined table of test performance and another for the best
hyperparameters of each outer fold.
'''

import os
import glob
import json
import argparse
import pandas as pd
from medl.settings import RESULTSDIR

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Output directory. Specifiy absolute path or'\
                                                    ' relative path under RESULTSDIR')
args = parser.parse_args()

strOutputDir = args.output_dir
if not strOutputDir.startswith(os.path.sep):
    strOutputDir = os.path.join(RESULTSDIR, strOutputDir)
    
if not os.path.exists(strOutputDir):
    raise NotADirectoryError(strOutputDir + ' does not exist')

lsModelDirs = glob.glob(os.path.join(strOutputDir, 'fold*', 'best_model'))
lsModelDirs.sort()

lsFoldNames = [x.split(os.path.sep)[-2] for x in lsModelDirs]

lsHyperParams = []
lsResults = []
for strModelDir in lsModelDirs:
    with open(os.path.join(strModelDir, 'params.json'), 'r') as f:
        lsHyperParams += [json.load(f)]
        
    df = pd.read_csv(os.path.join(strModelDir, 'final_test.csv'), index_col=0)
    lsResults += [df.loc[df['Partition'] == 'Val']]
    
dfResults = pd.concat(lsResults, axis=0)
dfResults.index = lsFoldNames
dfResults.to_csv(os.path.join(strOutputDir, 'test_results.csv'))
dfHyperParams = pd.DataFrame(lsHyperParams, index=lsFoldNames)
dfHyperParams.to_csv(os.path.join(strOutputDir, 'best_hyperparams.csv'))

print('Mean hyperparameters')
print(dfHyperParams.mean())

print('Mean performance')
print(dfResults.mean())