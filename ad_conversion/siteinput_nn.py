'''
Evaluate MLP w/ additional site input for predicting MCI conversion. Should be
run after conventional_nn_hpo.py, since the winning architectures from that HPO
search will be used. The additional site input will simply be concatenated to 
each of these architectures.
'''
import os

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import argparse
import json

from ray import tune

from medl.settings import RESULTSDIR

from ad_models import SiteInputModel
    
def _expand_dir(dir):
    # Expand output directory to absolute path if needed
    if not dir.startswith(os.path.sep):
        return os.path.join(RESULTSDIR, dir)
    else:
        return dir
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpo_dir', type=str, help='Directory containing HPO results for conventional NN. '\
                                                    'Specifiy absolute path or relative path under RESULTSDIR')
    parser.add_argument('--output_dir', type=str, help='Output directory. Specifiy absolute path or'\
                                                        ' relative path under RESULTSDIR')
    parser.add_argument('--folds', type=str, default='./10x10_kfolds_sitecluster.pkl',
                        help='Saved nested K-folds')
    parser.add_argument('--seed', type=int, default=3234, help='Random seed')
    args = parser.parse_args()
    
    strOutputDir = _expand_dir(args.output_dir)
    strHpoDir = _expand_dir(args.hpo_dir)
    
    if os.path.exists(strOutputDir):
        print('Warning: output directory already exists')
    else:
        os.makedirs(strOutputDir)

    lsHpoDirs = glob.glob(os.path.join(strHpoDir, 'fold*'))
    lsHpoDirs.sort()

    for strHpoDir in lsHpoDirs:
        # Find the best config from previous HPO
        analysis = tune.Analysis(strHpoDir)

        dictBestConfig = analysis.get_best_config(metric='Youden\'s index', mode='max')
        strBestLogdir = analysis.get_best_logdir(metric='Youden\'s index', mode='max')
        
        # Load trial results which contain # epochs trained
        with open(os.path.join(strBestLogdir, 'result.json'), 'r') as f:
            dictTrialResults = json.load(f)
            
        strFold = os.path.basename(strHpoDir)
        strFoldOutputDir = os.path.join(strOutputDir, strFold, 'best_model')
        os.makedirs(strFoldOutputDir, exist_ok=True)
        
        with open(os.path.join(strFoldOutputDir, 'params.json'), 'w') as f:
            json.dump(dictBestConfig, f, indent=4)
        
        model = SiteInputModel(dictBestConfig, strFoldOutputDir)
        model.final_test(int(1.1 * dictTrialResults['Epochs']))


    