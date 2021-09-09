'''
Run HPO to optimize an MLP architecture for predicting MCI conversion. 

Example usage:
python conventional_nn_hpo.py --output_dir </path/to/output/dir> --outer_fold <outer_fold_to_run>
'''

import os

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tkl

import ray
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch

from medl.settings import RESULTSDIR

from ad_models import BaseModel
    
class Trial(tune.Trainable): 
    # Wrapper layer needed for Tune's search API
            
    def setup(self, config):
        self.config = config
        self.model = BaseModel(self.config, self.logdir)
    
    def step(self):
        return self.model.cross_validate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Output directory. Specifiy absolute path or'\
                                                        ' relative path under RESULTSDIR')
    parser.add_argument('--outer_fold', type=int, help='Outer K-fold index')
    parser.add_argument('--folds', type=str, default='./10x10_kfolds_sitecluster.pkl',
                        help='Saved nested K-folds')
    parser.add_argument('--seed', type=int, default=3234, help='Random seed')
    parser.add_argument('--skip_hpo', action='store_true', help='Skip HPO and go to final model evaluation')
    args = parser.parse_args()

    # Define hyperparameter search space
    dictConfigSpace = {'layer_1_neurons': tune.uniform(8, 64),
                       'hidden_layers': tune.uniform(1, 8),
                       'last_layer_neurons': tune.uniform(4, 8),
                       'learning_rate': tune.uniform(1e-5, 1e-2),
                       'activation': tune.choice(['relu', 'elu', 'tanh']),
                       'dropout': tune.uniform(0.0, 0.75),
                       'outer_fold': args.outer_fold,
                       'folds': os.path.abspath(args.folds),
                       'seed': args.seed}
    
    # Expand output directory to absolute path if needed
    strOutputDir = args.output_dir
    if not strOutputDir.startswith(os.path.sep):
        strOutputDir = os.path.join(RESULTSDIR, strOutputDir)
    if os.path.exists(strOutputDir):
        print('Warning: output directory already exists')
    else:
        os.makedirs(strOutputDir)

    # Perform HPO using Scikit-optimize Bayesian optimizer
    if not args.skip_hpo:
        ray.init()

        suggester = SkOptSearch(metric='Youden\'s index', mode='max')

        tune.run(Trial, config=dictConfigSpace, search_alg=suggester,
                num_samples=100,
                resources_per_trial={'cpu': 16, 'gpu': 1}, 
                local_dir=strOutputDir,
                name=f'fold{args.outer_fold:02d}',
                stop={'training_iteration': 1},
                max_failures=4,
                verbose=3)

    # Find the best config and perform final evaluation
    analysis = tune.Analysis(os.path.join(strOutputDir, f'fold{args.outer_fold:02d}'))

    dictBestConfig = analysis.get_best_config(metric='Youden\'s index', mode='max')
    strBestLogdir = analysis.get_best_logdir(metric='Youden\'s index', mode='max')
    
    # Load trial results which contain # epochs trained
    with open(os.path.join(strBestLogdir, 'result.json'), 'r') as f:
        dictTrialResults = json.load(f)
    
    print('=========Best config=========')
    print(dictBestConfig)
    strFinalDir = os.path.join(strOutputDir, f'fold{args.outer_fold:02d}', 'best_model')
    os.makedirs(strFinalDir, exist_ok=True)
    
    # Save best config
    with open(os.path.join(strFinalDir, 'params.json'), 'w') as f:
        json.dump(dictBestConfig, f, indent=4)

    model = BaseModel(dictBestConfig, strFinalDir)
    model.final_test(int(1.1 * dictTrialResults['Epochs']))
    