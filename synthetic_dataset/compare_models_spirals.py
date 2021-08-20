'''
Compare several types of model on the spirals classification problem with random effects:

1. conventional neural network (multilevel perceptron)
2. conventional neural network with concatenated cluster membership input
3. mixed-effect neural network which models random effects
4. conventional neural network trained with meta-learning domain generalization (Li 2018)

Models are evaluated over a range of spiral parameters, varying the difficulty of the problem:

1. inter-cluster variance (random slope s.d.)
2. spiral length
3. data noise
4. number of classes

Parallelization is implemented using Ray Tune.

'''
SEED = 2578
OUTDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/spirals_20210719'

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import ray
from ray import tune

class Comparison(tune.Trainable):       
    
    def setup(self, config):
        sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL/')
        from spirals import make_spiral_random_slope, make_spiral_random_radius, plot_clusters
       
        np.random.seed(SEED)
        arrX, arrZ, arrY, arrRandomRadii = make_spiral_random_radius(10, 
                                                                     inter_cluster_sd=config['inter_cluster_sd'], 
                                                                     classes=config['classes'], 
                                                                     degrees=config['degrees'], 
                                                                     noise=config['noise'])

        self.arrX = arrX
        self.arrY = arrY
        self.arrZ = arrZ
        
        os.makedirs(self.logdir, exist_ok=True)

        fig, ax = plot_clusters(arrX, arrZ, arrY, arrRandomRadii)
        fig.savefig(os.path.join(self.logdir, 'clusters.png'))
        fig.savefig(os.path.join(self.logdir, 'clusters.svg'))
        plt.close(fig)
        
        np.savetxt(os.path.join(self.logdir, 'random_radii.txt'), arrRandomRadii)

    # def reset_config(self, new_config):
    #     self.setup(new_config)
    #     self.config = new_config

    def crossvalidate(self, model_fn, z_input=False, 
                      metalearn_training=False,
                      **kwargs):
        sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL/')
        from medl.models.metalearning import mldg
        
        dfResults = pd.DataFrame(index=range(self.arrZ.shape[1]), 
                                 columns=['Train', 'Test', 'Held-out group'])
            
        for iBatch in range(self.arrZ.shape[1]):
            arrXSeen = self.arrX[self.arrZ[:, iBatch] == 0, :]
            arrYSeen = self.arrY[self.arrZ[:, iBatch] == 0]
            arrZSeen = self.arrZ[self.arrZ[:, iBatch] == 0, :]
            arrZSeen = np.concatenate([arrZSeen[:, :iBatch], arrZSeen[:, (iBatch+1):]], axis=1)
            
            arrXUnseen = self.arrX[self.arrZ[:, iBatch] == 1, :]
            arrYUnseen = self.arrY[self.arrZ[:, iBatch] == 1,]
            arrZUnseen = np.zeros((arrXUnseen.shape[0], arrZSeen.shape[1]))
            
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=32)
            # Stratify split by batch membership and class
            arrStrat = np.array([str(y) + str(np.where(z)[0]) for y, z in zip(arrYSeen, arrZSeen)])
            
            arrTrainIdx, arrTestIdx = next(splitter.split(arrXSeen, arrStrat))
            arrXSeenTrain = arrXSeen[arrTrainIdx]
            arrYSeenTrain = arrYSeen[arrTrainIdx]
            arrZSeenTrain = arrZSeen[arrTrainIdx]
            arrXSeenTest = arrXSeen[arrTestIdx]
            arrYSeenTest = arrYSeen[arrTestIdx]
            arrZSeenTest = arrZSeen[arrTestIdx]
            
            tf.random.set_seed(89)
            if z_input:
                model = model_fn(arrZSeen.shape[1], inputs=2, outputs=arrYSeen.shape[1], **kwargs)
                inputsTrain = (arrXSeenTrain, arrZSeenTrain)
                inputsTest = (arrXSeenTest, arrZSeenTest)
                inputsUnseen = (arrXUnseen, arrZUnseen)
            else:
                model = model_fn(inputs=2, outputs=arrYSeen.shape[1], **kwargs)
                inputsTrain = arrXSeenTrain
                inputsTest = arrXSeenTest
                inputsUnseen = arrXUnseen
            
            # if metalearn_training:
            #     model = mldg(arrXSeenTrain, arrYSeenTrain, arrZSeenTrain, model, epochs=40)    
                
            # else:    
            model.fit(inputsTrain, arrYSeenTrain, batch_size=32, epochs=40, verbose=0)
            
            dfResults['Train'].loc[iBatch] = model.evaluate(inputsTrain, arrYSeenTrain, verbose=0)[1]
            dfResults['Test'].loc[iBatch] = model.evaluate(inputsTest, arrYSeenTest, verbose=0)[1]       
            dfResults['Held-out group'].loc[iBatch] = model.evaluate(inputsUnseen, arrYUnseen, verbose=0)[1]       
                            
        return dfResults

    def step(self):
        sys.path.append('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL/')
        from medl.models.mlp_classifiers import base_model, concat_model, me_model
        
        dfBaseModel = self.crossvalidate(base_model)
        dfBaseModel.to_csv(os.path.join(self.logdir, 'base_model.csv'))

        dfConcatModel = self.crossvalidate(concat_model, z_input=True)
        dfConcatModel.to_csv(os.path.join(self.logdir, 'concat_model.csv'))

        dfMEModel = self.crossvalidate(me_model, z_input=True, 
                                kl_weight=1/8000, prior_scale=0.5)
        dfMEModel.to_csv(os.path.join(self.logdir, 'me_model.csv'))

        # dfMetaLearnModel = self.crossvalidate(base_model, metalearn_training=True)
        # dfMetaLearnModel.to_csv(os.path.join(self.logdir, 'metalearn_model.csv'))
        
        return {'base_test': dfBaseModel['Test'].mean(),
                'base_heldout': dfBaseModel['Held-out group'].mean(),
                'concat_test': dfConcatModel['Test'].mean(),
                'concat_heldout': dfConcatModel['Held-out group'].mean(),
                'me_test': dfMEModel['Test'].mean(),
                'me_heldout': dfMEModel['Held-out group'].mean(),
                # 'mldg_test': dfMetaLearnModel['Test'].mean(),
                # 'mldg_heldout': dfMetaLearnModel['Held-out group'].mean(),
                }

ray.init()

dictConfig = {'classes': tune.grid_search([2, 3]),
              'inter_cluster_sd': tune.grid_search([0.1, 0.3, 0.5, 0.7, 1.0]),
              'noise': tune.grid_search([0.01, 0.05, 0.1, 0.2, 0.3]),
              'degrees': tune.grid_search([360, 540, 720])}

# dictConfig = {'classes': tune.grid_search([2]),
#               'inter_cluster_sd': tune.grid_search([0.3]),
#               'noise': tune.grid_search([0.2]),
#               'degrees': tune.grid_search([360])}

tune.run(Comparison, 
         config=dictConfig,
         metric='me_test', 
         mode='max', 
         local_dir=OUTDIR,
         stop={'training_iteration': 1}, 
         resources_per_trial={'gpu': 1}, 
         num_samples=1, 
         reuse_actors=False)
