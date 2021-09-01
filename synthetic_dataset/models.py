import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tkl
from sklearn.model_selection import StratifiedShuffleSplit

from medl.models.random_effects import RandomEffects
from medl.metrics import balanced_accuracy
from medl.models.metalearning import mldg

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
    if isinstance(n_features, tuple):
        n_features = n_features[0]
        
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

# def me_model(n_features, n_clusters, n_classes=2):
#     tInput = tkl.Input(n_features)
#     tInputZ = tkl.Input(n_clusters)
    
#     tRE = RandomEffects(n_features, post_loc_init_scale=0, post_scale_init_min=0.1, 
#                         post_scale_init_range=0.1, kl_weight=0.001, 
#                         prior_scale=1,
#                         name='re_slope')(tInputZ)
#     # tConcat = tkl.Concatenate(axis=-1)([tInput, tRE * tInput])
    
#     tDense1 = tkl.Dense(4, activation='relu')(tInput)
#     tDense2 = tkl.Dense(4, activation='relu')(tDense1)
#     tDense3 = tkl.Dense(4, activation='relu')(tDense2)
    
#     tConcat = tkl.Concatenate(axis=-1)([tDense3, tRE * tInput])
#     tOutput = tkl.Dense(n_classes, activation='softmax')(tConcat)
           
#     model = tf.keras.Model((tInput, tInputZ), tOutput)
#     model.compile(loss='categorical_crossentropy',
#                   metrics=[balanced_accuracy, tf.keras.metrics.AUC(curve='PR', name='auprc')],
#                   optimizer=tf.keras.optimizers.Adam())
    
#     return model

def cross_validate(model_fn, arrX, arrZ, arrY, cluster_input=False, test=False, mldg_training=False, epochs=100, seed=2):
    """Perform leave-one-cluster-out cross-validation

    Args:
        model_fn (function): model creation function
        arrX (np.array): input 
        arrZ (np.array): cluster membership matrix
        arrY (np.array): target
        cluster_input (bool, optional): whether the model takes a cluster membership 
        input. Defaults to False.
        test (bool, optional): evaluate on test partition and held-out cluster. 
        Defaults to False.
        epochs (int, optional): Training duration. Defaults to 100.
        seed (int, optional): Random seed. Defaults to 2.

    Returns:
        dataframe containing performance
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
        arrYSeen = arrY[arrZ[:, iCluster] == 0,]
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
            arrStratInner = arrStrat[arrTrainIdx,]
            splitterInner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
            
            arrTrainIdx, arrValIdx = next(splitterInner.split(arrXSeenTrain, arrStratInner))
            arrXSeenVal = arrXSeenTrain[arrValIdx]
            arrYSeenVal = arrYSeenTrain[arrValIdx]
            arrZSeenVal = arrZSeenTrain[arrValIdx]           
            arrXSeenTrain = arrXSeenTrain[arrTrainIdx]
            arrYSeenTrain = arrYSeenTrain[arrTrainIdx]
            arrZSeenTrain = arrZSeenTrain[arrTrainIdx]            
        
        tf.random.set_seed(seed)
        if cluster_input:
            model = model_fn(arrX.shape[1:], arrZSeenTrain.shape[1], n_classes=arrY.shape[1])
            inputsTrain = (arrXSeenTrain, arrZSeenTrain)
            inputsVal = (arrXSeenVal, arrZSeenVal)
            inputsUnseen = (arrXUnseen, arrZUnseen)
        else:
            model = model_fn(arrX.shape[1:], n_classes=arrY.shape[1])
            inputsTrain = arrXSeenTrain
            inputsVal = arrXSeenVal
            inputsUnseen = arrXUnseen

        if mldg_training:
            model = mldg(arrXSeenTrain, arrYSeenTrain, arrZSeenTrain, model, epochs=epochs)    
        else:
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

def validate(model_fn, arrX, arrZ, arrY, cluster_input=False, mldg_training=False, epochs=100, seed=2):
    """Perform single hold-out validation with 20% test split

    Args:
        model_fn (function): model creation function
        arrX (np.array): input 
        arrZ (np.array): cluster membership matrix
        arrY (np.array): target
        cluster_input (bool, optional): whether the model takes a cluster membership 
        input. Defaults to False.
        epochs (int, optional): Training duration. Defaults to 100.
        seed (int, optional): Random seed. Defaults to 2.

    Returns:
        model, test performance
    """    
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
        model = model_fn(arrX.shape[1:], arrZ.shape[1], n_classes=arrY.shape[1])
        inputsTrain = (arrXTrain, arrZTrain)
        inputsTest = (arrXTest, arrZTest)
    else:
        model = model_fn(arrX.shape[1:], n_classes=arrY.shape[1])
        inputsTrain = arrXTrain
        inputsTest = arrXTest
        
    if mldg_training:
        model = mldg(arrXTrain, arrYTrain, arrZTrain, model, epochs=epochs)
    else:        
        model.fit(inputsTrain, arrYTrain, batch_size=32, epochs=epochs, verbose=0)
        
    return model, model.evaluate(inputsTest, arrYTest, verbose=0)

def plot_percluster_decision_boundary(model, X, Y, Z, cluster_input=False, vmax=1.5, degrees=360, radii=None):
    """Plot decision boundaries learned per cluster. Currently only works with 2-class problems.

    Args:
        model (tf.keras.Model): trained model
        X (np.array): input
        Y (np.array): target
        Z (np.array): cluster membership matrix
        cluster_input (bool, optional): Whether model takes a cluster membership input. Defaults to False.
        vmax (float, optional): Extent of feature space to plot. Defaults to 1.5.
        degrees (int, optional): Spiral length (to plot true decision boundary). Defaults to 360.
        radii (array, optional): Cluster-specific radii (to plot true decision boundary). Defaults to None.

    Returns:
        figure, axes
    """    
    import matplotlib.pyplot as plt
    from spirals import make_spiral_true_boundary
    
    nClusters = Z.shape[1]
    nAxRows = int(np.ceil(nClusters / 5))
    fig, axes = plt.subplots(nAxRows, 5, figsize=(16, 3 * nAxRows), gridspec_kw={'hspace': 0.5})
    
    for iCluster in range(nClusters):
        ax = axes.flatten()[iCluster]
        
        arrXCluster = X[Z[:, iCluster]==1, :]
        arrYCluster = Y[Z[:, iCluster]==1, :]
        arrZCluster = Z[Z[:, iCluster]==1, :]
        
        # Create grid of points in feature space
        arrGridX1, arrGridX2 = np.mgrid[-vmax:vmax+0.1:0.1, -vmax:vmax+0.1:0.1]
        # Fill in extra confounded features with 0.5
        if X.shape[1] > 2:
            arrGridX3 = np.ones_like(arrGridX1) * 0.5
            arrGridXConf = np.stack([arrGridX3.flatten()] * (X.shape[1] - 2), axis=-1)
            arrGridX = np.concatenate([arrGridX1.reshape(-1, 1), arrGridX2.reshape(-1, 1), arrGridXConf], axis=1)
        else:    
            arrGridX = np.stack([arrGridX1.flatten(), arrGridX2.flatten()], axis=0).T
            
        if cluster_input:
            arrGridZ = np.zeros((arrGridX.shape[0], Z.shape[1]))
            arrGridZ[:, iCluster] = 1
            arrGridYFlat = model.predict((arrGridX, arrGridZ), verbose=0)
        else:
            arrGridYFlat = model.predict(arrGridX, verbose=0) 

        arrGridY = (arrGridYFlat[:, 1] >= 0.5).reshape(arrGridX1.shape).astype(int)
        
        # Use contour function to visualize decision boundary
        ax.contour(arrGridX1, arrGridX2, arrGridY, levels=1, colors='k')  
        ax.contourf(arrGridX1, arrGridX2, arrGridY, levels=1, colors=['C0', 'C1'], alpha=0.5)    
        
        # Plot data points
        ax.scatter(arrXCluster[arrYCluster[:, 0] == 1, 0], arrXCluster[arrYCluster[:, 0] == 1, 1], 
                   c='C0', s=5, alpha=0.9)
        ax.scatter(arrXCluster[arrYCluster[:, 1] == 1, 0], arrXCluster[arrYCluster[:, 1] == 1, 1], 
                   c='C1', s=10, alpha=0.9, marker='P')
        
        if radii is not None:
            # Plot true decision boundary (midpoint between classes)
            arrTrueBoundary = make_spiral_true_boundary(classes=Y.shape[1], degrees=degrees-180, 
                                                        radius=radii[iCluster])
            ax.plot(arrTrueBoundary[:1000, 0], 
                                        arrTrueBoundary[:1000, 1], 
                                        c='g', ls='--', lw=3)   
            ax.plot(arrTrueBoundary[1000:, 0], 
                                        arrTrueBoundary[1000:, 1], 
                                        c='g', ls='--', lw=3)
                
        # Include accuracy in subplot title    
        if cluster_input:
            acc = model.evaluate((arrXCluster, arrZCluster), arrYCluster, verbose=0)[1]
        else:
            acc = model.evaluate(arrXCluster, arrYCluster, verbose=0)[1]
        ax.set_title(f'Accuracy: {acc:.03f}')
        
        ax.set_xlim(-vmax, vmax)
        ax.set_ylim(-vmax, vmax)
        ax.set_aspect('equal')    
    return fig, axes
    