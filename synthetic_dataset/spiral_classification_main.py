#!/bin/python
'''
Main script for running conventional DL vs. MEDL comparison on the spiral
classification benchmark. 

Usage: python spiral_classification_main.py --output_dir
    /path/to/output/location

See python spiral_classification_main.py --help for other arguments which
control data generation. 

This script compares three models (implemented in ./models.py):

    1. Conventional MLP 
        * 3 hidden layers of 4 neurons each
    
    2. Conventional MLP with additional cluster membership input
        * cluster membership is one-hot encoded and concatenated to the 
        input features
        
    3. Mixed effects MLP

Leave-one-cluster-out cross-validation is conducted with 10% validation 
hold-out, 20% test hold-out. If the --evaluate_test flag is given, test 
and held-out cluster performance are returned, otherwise only the 
validation performance is returned. 

Outputs:

    * Long-form dataframe containing LOCO balanced accuracy for the 
    three models. 
    
    * Accuracy barplots
    
    * Scatterplots of each cluster's data points. 
    
    * JSON file containing all arguments
    
    * List of cluster-specific class ratios if simulating confounded 
    variables
'''

import sys, os
sys.path.append('../')
import argparse
import json

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

from spirals import make_spiral_random_radius, make_spiral_random_radius_confounder, plot_clusters


def _run(args):
    from medl.tfutils import set_gpu
    from models import base_model, concat_model, me_model, cross_validate

    if args.gpu:
        set_gpu(int(args.gpu), mem_frac=args.gpu_mem_frac)

    if os.path.exists(args.output_dir):
        print('Output directory exists', flush=True)
    else:
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'parameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    np.random.seed(args.random_seed)
    if args.confound_sd > 0:
        arrX, arrZ, arrY, arrRadii, arrRatio = make_spiral_random_radius_confounder(args.clusters,
                                                                                    mean_radius=0 if args.false_negatives else 1,
                                                                                    ratio_sd=args.confound_sd,
                                                                                    radius_sd=args.radius_sd,
                                                                                    degrees=args.degrees,
                                                                                    confounders=args.confounded_vars,
                                                                                    noise=args.noise)
        np.savetxt(os.path.join(args.output_dir, 'class_ratio.txt'), arrRatio)
    else:    
        arrX, arrZ, arrY, arrRadii = make_spiral_random_radius(args.clusters, 
                                                            mean_radius=0 if args.false_negatives else 1, 
                                                            inter_cluster_sd=args.radius_sd, 
                                                            degrees=args.degrees,
                                                            classes=args.classes,
                                                            noise=args.noise)
   
    figData, axData = plot_clusters(arrX, arrZ, arrY, arrRadii)
    figData.savefig(os.path.join(args.output_dir, 'clusters.svg'))
    figData.show()

    dfBase = cross_validate(base_model, arrX, arrZ, arrY, test=args.evaluate_test, 
        epochs=args.epochs, seed=args.random_seed)
    print('Conventional model', flush=True)
    print(dfBase.mean(), flush=True)
    
    dfConcat = cross_validate(concat_model, arrX, arrZ, arrY, cluster_input=True, 
                            test=args.evaluate_test, epochs=args.epochs, seed=args.random_seed)
    print('Conventional model w/ cluster input', flush=True)
    print(dfConcat.mean(), flush=True)
    
    dfME = cross_validate(me_model, arrX, arrZ, arrY, cluster_input=True, 
                        test=args.evaluate_test, epochs=args.epochs, seed=args.random_seed)
    print('ME-MLP', flush=True)
    print(dfME.mean(), flush=True)

    dfBaseLong = pd.melt(dfBase, var_name='Partition', value_name='Accuracy')
    dfBaseLong['Model'] = 'Conventional'
    dfConcatLong = pd.melt(dfConcat, var_name='Partition', value_name='Accuracy')
    dfConcatLong['Model'] = 'Cluster input'
    dfMELong = pd.melt(dfME, var_name='Partition', value_name='Accuracy')
    dfMELong['Model'] = 'ME-MLP'

    dfAll = pd.concat([dfBaseLong, dfConcatLong, dfMELong], axis=0)
    dfAll.to_csv(os.path.join(args.output_dir, 'results.csv'))

    figAcc, axAcc = plt.subplots()
    if args.evaluate_test:
        sns.barplot(data=dfAll, x='Partition', hue='Model', y='Accuracy', ax=axAcc, 
                    order=['Test', 'Held-out cluster'],
                    hue_order=['Conventional', 'Cluster input', 'ME-MLP'])
        
    else:
        sns.barplot(data=dfAll, x='Partition', hue='Model', y='Accuracy', ax=axAcc, 
                    order=['Train', 'Val'],
                    hue_order=['Conventional', 'Cluster input', 'ME-MLP'])
        
    figAcc.savefig(os.path.join(args.output_dir, 'accuracy.svg'))
    figAcc.show()
    
    return dfAll


def main(arg_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters', default=10, type=int, help='Number of clusters')
    parser.add_argument('--radius_sd', default=0.2, type=float, help='s.d. of cluster-specific random radii')
    parser.add_argument('--confound_sd', default=0, type=float, help='Add confounding effect. This arg determines class ratio (p_j) s.d.')
    parser.add_argument('--confounded_vars', default=2, type=int, help='Number of confounded variables to add')
    parser.add_argument('--false_negatives', action='store_true', help='Center random radii at 0 to create a Simpson\'s paradox situation')
    parser.add_argument('--noise', default=0.2, type=float, help='s.d. of added Gaussian noise')
    parser.add_argument('--degrees', default=360, type=int, help='Spiral length in degrees')
    parser.add_argument('--classes', default=2, type=int, help='Number of spirals')

    parser.add_argument('--evaluate_test', action='store_true', help='Evaluate on test partition and held-out clusters')
    parser.add_argument('--epochs', default=100, type=int, help='Training duration for all models')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')

    parser.add_argument('--gpu', default=None, help='GPU to use')
    parser.add_argument('--gpu_mem_frac', default=1.0, type=float, help='GPU memory %% limit')
    parser.add_argument('--random_seed', default=8, type=int, help='Random seed')

    args = parser.parse_args(arg_list)
    return _run(args)
    
    
if __name__ == '__main__':
    main()