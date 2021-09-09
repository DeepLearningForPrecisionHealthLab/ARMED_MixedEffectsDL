import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def _get_rows(x, indices):
    if isinstance(x, pd.DataFrame) | isinstance(x, pd.Series):
        return x.iloc[indices]
    else:
        return x[indices,]

class BasicKFoldUtil:
    def __init__(self, x, z, y, 
                 n_folds: int=5, 
                 kfold_class=StratifiedKFold, 
                 seed=8):
        
        self.x = x
        self.z = z
        self.y = y
        self.n_folds = n_folds
        self.seed = seed
        
        self.splitter = kfold_class(n_splits=n_folds, shuffle=True, random_state=seed)
        
        self.folds = self.create_folds()
        
    def create_folds(self) -> list:
        lsFolds = []

        # Stratify by cluster and target
        zvals = self.z.values if isinstance(self.z, pd.DataFrame) else self.z
        lsStrat = ['{}_{}'.format(label, cluster) for label, cluster in zip(self.y, zvals)]
        arrStrat = np.array(lsStrat)

        for arrTrainIdx, arrValIdx in self.splitter.split(self.x, arrStrat):
            assert len(np.intersect1d(arrTrainIdx, arrValIdx)) == 0 
            
            xTrain = _get_rows(self.x, arrTrainIdx)
            zTrain = _get_rows(self.z, arrTrainIdx)
            yTrain = _get_rows(self.y, arrTrainIdx)
            
            xVal = _get_rows(self.x, arrValIdx)
            zVal = _get_rows(self.z, arrValIdx)
            yVal = _get_rows(self.y, arrValIdx)
            
            lsFolds += [(xTrain, zTrain, yTrain, xVal, zVal, yVal)]
            
        return lsFolds
    
    def get_fold(self, idx):
        return self.folds[idx]
    
    
class NestedKFoldUtil(BasicKFoldUtil):
    def __init__(self, x, z, y, 
                 n_folds_outer: int=5, n_folds_inner: int=5,
                 kfold_class=StratifiedKFold, seed=8):
        self.n_folds_inner = n_folds_inner
        self.kfold_class = kfold_class
        super().__init__(x, z, y, n_folds=n_folds_outer, kfold_class=kfold_class, seed=seed)
           
    def create_folds(self) -> list:       
        lsFoldsOuter = super().create_folds()
        
        lsFolds = []
        
        for tupFoldsOuter in lsFoldsOuter:
            xTrain, zTrain, yTrain, xTest, zTest, yTest = tupFoldsOuter
            
            kfoldsInner = BasicKFoldUtil(xTrain, zTrain, yTrain, self.n_folds_inner,
                                         self.kfold_class, self.seed)
            
            lsFolds += [{'outer': tupFoldsOuter, 'inner': kfoldsInner}]
            
        return lsFolds
    
    def get_fold(self, idx_outer, idx_inner=None):
        if idx_inner:
            return self.folds[idx_outer]['inner'].get_fold(idx_inner)
        else:
            return self.folds[idx_outer]['outer']