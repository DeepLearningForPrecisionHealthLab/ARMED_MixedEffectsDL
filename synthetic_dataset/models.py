import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tkl
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append('../')
from medl.models.random_effects2 import RandomEffects
from medl.metrics import balanced_accuracy

def base_model(n_features, n_classes=2):
    tInput = tkl.Input(n_features)
    tDense1 = tkl.Dense(4, activation='relu')(tInput)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tDense3 = tkl.Dense(4, activation='relu')(tDense2)
    tOutput = tkl.Dense(n_classes, activation='softmax')(tDense3)
    
    model = tf.keras.Model(tInput, tOutput)
    model.compile(loss='categorical_crossentropy',
                  metrics=[balanced_accuracy, tf.keras.metrics.AUC(curve='PR', name='auprc')],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def concat_model(n_features, n_clusters, n_classes=2):
    tInput = tkl.Input(n_features)
    tInputZ = tkl.Input(n_clusters)
    
    tConcat = tkl.Concatenate(axis=-1)([tInput, tInputZ])
    
    tDense1 = tkl.Dense(4, activation='relu')(tConcat)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tDense3 = tkl.Dense(4, activation='relu')(tDense2)
    tOutput = tkl.Dense(n_classes, activation='softmax')(tDense3)
    
    model = tf.keras.Model((tInput, tInputZ), tOutput)
    model.compile(loss='categorical_crossentropy',
                  metrics=[balanced_accuracy, tf.keras.metrics.AUC(curve='PR', name='auprc')],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def me_model(n_features, n_clusters, n_classes=2):
    tInput = tkl.Input(n_features)
    tInputZ = tkl.Input(n_clusters)
    
    tRE = RandomEffects(n_features, post_loc_init_scale=0, post_scale_init_min=0.1, 
                        post_scale_init_range=0.1, kl_weight=1/8000, 
                        l1_weight=0.001,
                        prior_scale=0.5,
                        name='re_slope')(tInputZ)
    # tConcat = tkl.Concatenate(axis=-1)([tInput, tRE * tInput])
    
    tDense1 = tkl.Dense(4, activation='relu')(tInput)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tDense3 = tkl.Dense(4, activation='relu')(tDense2)
    
    tDenseRE = tkl.Dense(4, activation='relu')(tRE * tInput)
    tDenseRE2 = tkl.Dense(4, activation='relu')(tDenseRE)
    tConcat = tkl.Concatenate(axis=-1)([tDense3, tDenseRE2])
    tDenseMixed = tkl.Dense(4, activation='relu')(tConcat)
    tOutput = tkl.Dense(n_classes, activation='softmax')(tDenseMixed)
           
    model = tf.keras.Model((tInput, tInputZ), tOutput)
    model.compile(loss='categorical_crossentropy',
                  metrics=[balanced_accuracy, tf.keras.metrics.AUC(curve='PR', name='auprc')],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def cross_validate(model_fn, arrX, arrZ, arrY, cluster_input=False, test=False, epochs=100, seed=2):
    """Perform leave-one-cluster-out cross-validation

    Args:
        test (bool, optional): Perform final evaluation on test partition. Defaults to False.

    Returns:
        dict: mean accuracy for train, val, held-out clusters
    """        
    if test: 
        lsColumns = ['Train', 'Test', 'Held-out cluster']
    else:
        lsColumns = ['Train', 'Val']
        
    dfResults = pd.DataFrame(index=range(arrZ.shape[1]), 
                                columns=lsColumns)
        
    for iCluster in range(arrZ.shape[1]):
        # Separate held-out cluster from rest of data
        arrXSeen = arrX[arrZ[:, iCluster] == 0, :]
        arrYSeen = arrY[arrZ[:, iCluster] == 0]
        arrZSeen = arrZ[arrZ[:, iCluster] == 0, :]
        arrZSeen = np.concatenate([arrZSeen[:, :iCluster], arrZSeen[:, (iCluster+1):]], axis=1)
        
        arrXUnseen = arrX[arrZ[:, iCluster] == 1, :]
        arrYUnseen = arrY[arrZ[:, iCluster] == 1,]
        arrZUnseen = np.zeros((arrXUnseen.shape[0], arrZSeen.shape[1]))
        
        # Split into train/test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
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
            splitterInner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
            
            arrTrainIdx, arrValIdx = next(splitterInner.split(arrXSeenTrain, arrStratInner))
            arrXSeenTrain = arrXSeen[arrTrainIdx]
            arrYSeenTrain = arrYSeen[arrTrainIdx]
            arrZSeenTrain = arrZSeen[arrTrainIdx]
            arrXSeenVal = arrXSeen[arrValIdx]
            arrYSeenVal = arrYSeen[arrValIdx]
            arrZSeenVal = arrZSeen[arrValIdx]           
        
        tf.random.set_seed(seed)
        if cluster_input:
            model = model_fn(arrX.shape[1], arrZSeenTrain.shape[1], n_classes=arrY.shape[1])
            inputsTrain = (arrXSeenTrain, arrZSeenTrain)
            inputsVal = (arrXSeenVal, arrZSeenVal)
            inputsUnseen = (arrXUnseen, arrZUnseen)
        else:
            model = model_fn(arrX.shape[1], n_classes=arrY.shape[1])
            inputsTrain = arrXSeenTrain
            inputsVal = arrXSeenVal
            inputsUnseen = arrXUnseen

        log = model.fit(inputsTrain, arrYSeenTrain, 
                        validation_data=(inputsVal, arrYSeenVal), 
                        batch_size=32, epochs=epochs, verbose=0)
                
        # Gather accuracy values
        dfResults['Train'].loc[iCluster] = model.evaluate(inputsTrain, arrYSeenTrain, verbose=0)[1]
        
        if test:
            dfResults['Test'].loc[iCluster] = model.evaluate(inputsVal, arrYSeenVal, verbose=0)[1]       
            dfResults['Held-out cluster'].loc[iCluster] = model.evaluate(inputsUnseen, arrYUnseen, verbose=0)[1]
        else:
            dfResults['Val'].loc[iCluster] = model.evaluate(inputsVal, arrYSeenVal, verbose=0)[1]       
        
        del model
        
    return dfResults

def validate(model_fn, arrX, arrZ, arrY, cluster_input=False, epochs=100, seed=2):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=32)
    # Stratify split by batch membership and class
    arrStrat = np.array([str(y) + str(np.where(z)[0]) for y, z in zip(arrY, arrZ)])

    arrTrainIdx, arrTestIdx = next(splitter.split(arrX, arrStrat))
    arrXTrain = arrX[arrTrainIdx]
    arrYTrain = arrY[arrTrainIdx]
    arrZTrain = arrZ[arrTrainIdx]
    arrXTest = arrX[arrTestIdx]
    arrYTest = arrY[arrTestIdx]
    arrZTest = arrZ[arrTestIdx]
    
    tf.random.set_seed(seed)
    if cluster_input:
        model = model_fn(arrX.shape[1], arrZTrain.shape[1])
        inputsTrain = (arrXTrain, arrZTrain)
        inputsTest = (arrXTest, arrZTest)
    else:
        model = model_fn(arrX.shape[1])
        inputsTrain = arrXTrain
        inputsTest = arrXTest
        
    model.fit(arrXTrain, arrYTrain, batch_size=32, epochs=epochs, verbose=0)
    return model, model.evaluate(arrXTrain, arrYTrain)