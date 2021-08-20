SEED = 1267
OUTDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/spirals_confound_20210819'

CONFOUND_SD = 0.5
RE_SD = 0.0
CONFOUNDERS = 2

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tkl

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

import ray
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch

from spirals import make_spiral_random_radius_confounder

def plot_decision_boundary(model, X, Y, Z=None, ax=None, vmax=2):       
    arrGridX1, arrGridX2 = np.mgrid[-vmax:vmax+0.1:0.1, -vmax:vmax+0.1:0.1]
    arrGridXConf = np.ones((arrGridX1.size, X.shape[1])) * 0.5
    arrGridX = np.concatenate([arrGridX1.reshape(-1, 1), arrGridX2.reshape(-1, 1), arrGridXConf], axis=1)
    
    if Z is not None:
        arrGridZ = np.zeros((arrGridX.shape[0], Z.shape[1]))
        arrGridZ[:, Z.sum(axis=0).argmax()] = 1
        arrGridYFlat = model.predict((arrGridX, arrGridZ), verbose=0)
    else:
        arrGridYFlat = model.predict(arrGridX, verbose=0) 

    arrGridY = (arrGridYFlat[:, 1] >= 0.5).reshape(arrGridX1.shape).astype(int)
    
    if ax is None:
        ax = plt.gca()
        
    ax.contour(arrGridX1, arrGridX2, arrGridY, levels=1, colors='k')  
    ax.contourf(arrGridX1, arrGridX2, arrGridY, levels=1, colors=['C0', 'C1'], alpha=0.5)    
    ax.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c='C0', s=5, alpha=0.9)
    ax.scatter(X[Y[:, 1] == 1, 0], X[Y[:, 1] == 1, 1], c='C1', s=10, alpha=0.9, marker='P')
        
    return fig, ax

class Model:
    def __init__(self, config, logdir, plot=False):
        sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL/')
        from spirals import plot_clusters
        
        self.config = config
        self.logdir = logdir
                
        np.random.seed(SEED)
        arrX, arrZ, arrY, arrRadii, arrRatio = make_spiral_random_radius_confounder(10, 
                                                                                    radius_sd=RE_SD, 
                                                                                    ratio_sd=CONFOUND_SD,
                                                                                    degrees=360, 
                                                                                    noise=0.2,
                                                                                    confounders=CONFOUNDERS)
        
        self.arrX = arrX
        self.arrY = arrY
        self.arrZ = arrZ
        
        os.makedirs(self.logdir, exist_ok=True)

        if plot:
            fig, ax = plot_clusters(arrX, arrZ, arrY, arrRadii, {'classes': 2, 'degrees': 360})
            fig.savefig(os.path.join(self.logdir, 'clusters.png'))
            fig.savefig(os.path.join(self.logdir, 'clusters.svg'))
            plt.close(fig)
            
            np.savetxt(os.path.join(self.logdir, 'class_ratio.txt'), arrRatio)
        
    def create_model(self, config): 
        from medl.metrics import balanced_accuracy
        from medl.models.random_effects2 import RandomEffects
             
        tInput = tkl.Input(self.arrX.shape[1])
        tInputZ = tkl.Input(self.arrZ.shape[1] - 1)
        
        tRE = RandomEffects(self.arrX.shape[1], 
                            post_loc_init_scale=0, 
                            post_scale_init_min=config['post_scale_init_min'], 
                            post_scale_init_range=config['post_scale_init_range'], 
                            kl_weight=config['kl_weight'], 
                            l1_weight=config['l1_weight'],
                            prior_scale=config['prior_scale'],
                            name='re_slope')(tInputZ)
        
        tDense1 = tkl.Dense(4, activation='relu')(tInput)
        tDense2 = tkl.Dense(4, activation='relu')(tDense1)
        tDense3 = tkl.Dense(4, activation='relu')(tDense2)
        
        re = tRE * tInput
        re = tkl.Dense(int(config['dense_re_1_neurons']), activation='relu')(re)
        if config['dense_re_2_neurons'] >= 1:
            re = tkl.Dense(int(config['dense_re_2_neurons']), activation='relu')(re)
        if config['dense_re_3_neurons'] >= 1:
            re = tkl.Dense(int(config['dense_re_3_neurons']), activation='relu')(re)
            
        mixed = tkl.Concatenate(axis=-1)([tDense3, re])
        
        if config['dense_mixed_neurons'] >= 1:
            mixed = tkl.Dense(int(config['dense_mixed_neurons']), activation='relu')(mixed)
            
        tOutput = tkl.Dense(2, activation='softmax')(mixed)
            
        model = tf.keras.Model((tInput, tInputZ), tOutput)
        model.compile(loss='categorical_crossentropy',
                      metrics=[balanced_accuracy, tf.keras.metrics.AUC(curve='PR', name='auprc')],
                      optimizer=tf.keras.optimizers.Adam())
        
        return model
        
    def cross_validate(self, test=False):
        if test: 
            lsColumns = ['Train', 'Test', 'Held-out group']
        else:
            lsColumns = ['Train', 'Val']
            
        dfResults = pd.DataFrame(index=range(self.arrZ.shape[1]), 
                                 columns=lsColumns)
        
        fig, ax = plt.subplots(5, 2, figsize=(6, 15))
            
        for iCluster in range(self.arrZ.shape[1]):
            arrXSeen = self.arrX[self.arrZ[:, iCluster] == 0, :]
            arrYSeen = self.arrY[self.arrZ[:, iCluster] == 0]
            arrZSeen = self.arrZ[self.arrZ[:, iCluster] == 0, :]
            arrZSeen = np.concatenate([arrZSeen[:, :iCluster], arrZSeen[:, (iCluster+1):]], axis=1)
            
            arrXUnseen = self.arrX[self.arrZ[:, iCluster] == 1, :]
            arrYUnseen = self.arrY[self.arrZ[:, iCluster] == 1,]
            arrZUnseen = np.zeros((arrXUnseen.shape[0], arrZSeen.shape[1]))
            
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            # Stratify split by batch membership and class
            arrStrat = np.array([str(y) + str(np.where(z)[0]) for y, z in zip(arrYSeen, arrZSeen)])
            
            arrTrainIdx, arrValIdx = next(splitter.split(arrXSeen, arrStrat))
            arrXSeenTrain = arrXSeen[arrTrainIdx]
            arrYSeenTrain = arrYSeen[arrTrainIdx]
            arrZSeenTrain = arrZSeen[arrTrainIdx]
            arrXSeenVal = arrXSeen[arrValIdx]
            arrYSeenVal = arrYSeen[arrValIdx]
            arrZSeenVal = arrZSeen[arrValIdx]
            
            if not test:
                # Split train again into train and validation
                arrStratInner = arrStrat[arrTrainIdx]
                splitterInner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
                
                arrTrainIdx, arrValIdx = next(splitterInner.split(arrXSeenTrain, arrStratInner))
                arrXSeenTrain = arrXSeen[arrTrainIdx]
                arrYSeenTrain = arrYSeen[arrTrainIdx]
                arrZSeenTrain = arrZSeen[arrTrainIdx]
                arrXSeenVal = arrXSeen[arrValIdx]
                arrYSeenVal = arrYSeen[arrValIdx]
                arrZSeenVal = arrZSeen[arrValIdx]           
            
            tf.random.set_seed(SEED)
            model = self.create_model(self.config)
            inputsTrain = (arrXSeenTrain, arrZSeenTrain)
            inputsVal = (arrXSeenVal, arrZSeenVal)
            inputsUnseen = (arrXUnseen, arrZUnseen)

            log = model.fit(inputsTrain, arrYSeenTrain, 
                            validation_data=(inputsVal, arrYSeenVal), 
                            batch_size=32, epochs=100, verbose=0)
            for val in ['balanced_accuracy', 'val_balanced_accuracy']:
                ax.flatten()[iCluster].plot(log.history[val], label=val)
            ax.flatten()[iCluster].legend()
            
            dfResults['Train'].loc[iCluster] = model.evaluate(inputsTrain, arrYSeenTrain, verbose=0)[1]
            
            if test:
                dfResults['Test'].loc[iCluster] = model.evaluate(inputsVal, arrYSeenVal, verbose=0)[1]       
                dfResults['Held-out group'].loc[iCluster] = model.evaluate(inputsUnseen, arrYUnseen, verbose=0)[1]
            else:
                dfResults['Val'].loc[iCluster] = model.evaluate(inputsVal, arrYSeenVal, verbose=0)[1]       
            
            del model
            
        dfResults.to_csv(os.path.join(self.logdir, 'loco_accuracy.csv'))
        fig.savefig(os.path.join(self.logdir, 'training_curve.svg'))
            
        if test: 
            return {'train_acc': dfResults['Train'].mean(),
                    'test_acc': dfResults['Test'].mean(),
                    'heldout_acc': dfResults['Held-out group'].mean()}
        else: 
            return {'train_acc': dfResults['Train'].mean(),
                    'val_acc': dfResults['Val'].mean()}

class Trial(tune.Trainable): 
            
    def setup(self, config):
        self.config = config
        self.model = Model(self.config, self.logdir, plot=False)
    
    def step(self, test=False):
        return self.model.cross_validate()
 
            
dictConfigSpace = {'post_scale_init_min': tune.uniform(0.01, 1),
                   'post_scale_init_range': tune.uniform(0.1, 0.5),
                   'kl_weight': tune.uniform(1e-4, 1e-1),
                   'l1_weight': tune.uniform(1e-4, 1e-1),
                   'prior_scale': tune.uniform(0.1, 3),
                   'dense_re_1_neurons': tune.quniform(1, 8, 1.0),
                   'dense_re_2_neurons': tune.quniform(0, 8, 1.0),
                   'dense_re_3_neurons': tune.quniform(0, 8, 1.0),
                   'dense_mixed_neurons': tune.quniform(0, 8, 1.0)}

ray.init()

suggester = SkOptSearch(metric='val_acc', mode='max')

tune.run(Trial, config=dictConfigSpace, search_alg=suggester,
        num_samples=100,
        resources_per_trial={'cpu': 16, 'gpu': 1}, 
        local_dir=OUTDIR,
        name='run1',
        stop={'training_iteration': 1},
        max_failures=4,
        verbose=3)

analysis = tune.Analysis(os.path.join(OUTDIR, 'run1'))

dictBestConfig = analysis.get_best_config(metric='val_acc', mode='max')
strFinalDir = os.path.join(OUTDIR, 'run1', 'best_model')
os.makedirs(strFinalDir, exist_ok=True)

model = Model(dictBestConfig, strFinalDir, plot=True)
dictFinalResults = model.cross_validate(test=True)
print(dictFinalResults)