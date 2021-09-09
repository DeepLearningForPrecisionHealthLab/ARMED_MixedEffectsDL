import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as tkl
from medl.models.random_effects import RandomEffects
from medl.crossvalidation.splitting import NestedKFoldUtil
from medl.metrics import classification_metrics

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def simple_nn_classifier(n_features):
    tInput = tkl.Input(n_features)
    tDense1 = tkl.Dense(4, activation='relu')(tInput)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tOutput = tkl.Dense(1, activation='sigmoid')(tDense2)
    
    model = tf.keras.Model(tInput, tOutput)
    model.compile(loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def simple_me_nn_classifier(n_features, n_clusters):
    tInput = tkl.Input(n_features)
    tInputZ = tkl.Input(n_clusters)
    
    tRE = RandomEffects(4, post_loc_init_scale=0, post_scale_init_min=0.1, 
                        post_scale_init_range=0.1, kl_weight=0.001, 
                        l1_weight=0.,
                        prior_scale=0.5,
                        name='re_slope')(tInputZ)
    
    tDense1 = tkl.Dense(4, activation='relu')(tInput)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tOutput = tkl.Dense(1, activation='sigmoid')(tDense2 + (tDense2 * tRE))
    
    model = tf.keras.Model((tInput, tInputZ), tOutput)
    model.compile(loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model


class BaseModel:
    def __init__(self, config, logdir):
        """Model wrapper for use with Tune trials.

        Args:
            config (dict): hyperparameter provided by Tune search algorithm
            logdir (str): output path
        """        
                
        self.config = config
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        
        with open(config['folds'], 'rb') as f:
            self.kfolds: NestedKFoldUtil = pickle.load(f) 
                
    def create_model(self, config) -> tf.keras.Model: 
        """Generates an MLP model

        Args:
            config (dict): hyperparameters

        Returns:
            tf.keras.Model: model
        """
        tf.get_logger().setLevel('ERROR')
             
        tInput = tkl.Input(self.kfolds.x.shape[1], name='input')
        
        tX = tInput
        
        # Round from float to int (since SkOpt doesn't sample integers)
        nNeuronsFirst = int(config['layer_1_neurons'])
        nNeuronsLast = int(config['last_layer_neurons'])
        nLayers = int(config['hidden_layers'])
        
        taper = np.power(nNeuronsLast / nNeuronsFirst, 1 / nLayers)
        
        for iLayer in range(nLayers):
            if iLayer == (nLayers - 1):
                neurons = nNeuronsLast
            neurons = int(nNeuronsFirst * (taper ** iLayer))
            
            tX = tkl.Dense(neurons, activation=config['activation'], name=f'dense{iLayer}')(tX)
        
        tX = tkl.Dropout(config['dropout'], name='dropout')(tX)
                    
        tOutput = tkl.Dense(1, activation='sigmoid')(tX)
            
        model = tf.keras.Model(tInput, tOutput)
        model.compile(loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                      optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
        return model
    
    def cross_validate(self):
        lsResults = []
        nInnerFolds = self.kfolds.n_folds_inner
        
        for iFold in range(nInnerFolds):
            dfXTrain, _, dfYTrain, dfXVal, _, dfYVal = self.kfolds.get_fold(self.config['outer_fold'], 
                                                                            idx_inner=iFold)
            
            scaler = StandardScaler()
            imputer = SimpleImputer()
            
            arrXTrain = scaler.fit_transform(dfXTrain)
            arrXTrain = imputer.fit_transform(arrXTrain)
            arrXVal = scaler.transform(dfXVal)
            arrXVal = imputer.transform(arrXVal)
            
            tf.random.set_seed(self.config['seed'])
            model: tf.keras.Model = self.create_model(self.config)
            dictClassWeights = {0: dfYTrain.mean(), 1: (1 - dfYTrain.mean())}
            
            log = model.fit(arrXTrain, dfYTrain,
                            validation_data=(arrXVal, dfYVal),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auprc', patience=10)],
                            epochs=100,
                            verbose=0,
                            class_weight=dictClassWeights)
            nTrainedEpochs = len(log.history['loss']) - 10
            
            arrYPredTrain = model.predict(arrXTrain)
            dictMetricsTrain, youden = classification_metrics(dfYTrain, arrYPredTrain)
            dictMetricsTrain.update(Partition='Train', Epochs=nTrainedEpochs)
            
            arrYPredVal = model.predict(arrXVal)
            dictMetricsVal, _ = classification_metrics(dfYVal, arrYPredVal, youden)
            dictMetricsVal.update(Partition='Val', Epochs=nTrainedEpochs)
            
            lsResults += [dictMetricsTrain, dictMetricsVal]
            
        dfResults = pd.DataFrame(lsResults)
        dfResults.to_csv(os.path.join(self.logdir, 'inner_crossval.csv'))        
        
        # dfResultsTrain = dfResults.loc[dfResults['Partition'] == 'Train']    
        dfResultsVal = dfResults.loc[dfResults['Partition'] == 'Val']
        
        return dfResultsVal.mean().to_dict()
    
    def final_test(self, epochs):
        dfXTrain, _, dfYTrain, dfXVal, _, dfYVal = self.kfolds.get_fold(self.config['outer_fold'])
        
        scaler = StandardScaler()
        imputer = SimpleImputer()
        
        arrXTrain = scaler.fit_transform(dfXTrain)
        arrXTrain = imputer.fit_transform(arrXTrain)
        arrXVal = scaler.transform(dfXVal)
        arrXVal = imputer.transform(arrXVal)
        
        tf.random.set_seed(self.config['seed'])
        model: tf.keras.Model = self.create_model(self.config)
        dictClassWeights = {0: dfYTrain.mean(), 1: (1 - dfYTrain.mean())}
        
        log = model.fit(arrXTrain, dfYTrain,
                        validation_data=(arrXVal, dfYVal),
                        epochs=epochs,
                        verbose=0,
                        class_weight=dictClassWeights)
        model.save(self.logdir)
        
        arrYPredTrain = model.predict(arrXTrain)
        dictMetricsTrain, youden = classification_metrics(dfYTrain, arrYPredTrain)
        dictMetricsTrain.update(Partition='Train')
        
        arrYPredVal = model.predict(arrXVal)
        dictMetricsVal, _ = classification_metrics(dfYVal, arrYPredVal, youden)
        dictMetricsVal.update(Partition='Val')
            
        dfResults = pd.DataFrame([dictMetricsTrain, dictMetricsVal])
        dfResults.to_csv(os.path.join(self.logdir, 'final_test.csv'))        
        
        # dfResultsTrain = dfResults.loc[dfResults['Partition'] == 'Train']    
        dfResultsVal = dfResults.loc[dfResults['Partition'] == 'Val']
        
        print(dfResultsVal.mean())
        return


class SiteInputModel(BaseModel):
                
    def create_model(self, config) -> tf.keras.Model: 
        """Generates an ME-MLP model

        Args:
            config (dict): hyperparameters

        Returns:
            tf.keras.Model: model
        """
        tf.get_logger().setLevel('ERROR')
             
        tInput = tkl.Input(self.kfolds.x.shape[1], name='input_x')
        tInputZ = tkl.Input(self.kfolds.z.shape[1], name='input_z')
        tX = tkl.Concatenate(axis=-1)([tInput, tInputZ])
        
        # Round from float to int (since SkOpt doesn't sample integers)
        nNeuronsFirst = int(config['layer_1_neurons'])
        nNeuronsLast = int(config['last_layer_neurons'])
        nLayers = int(config['hidden_layers'])
        
        taper = np.power(nNeuronsLast / nNeuronsFirst, 1 / nLayers)
        
        for iLayer in range(nLayers):
            if iLayer == (nLayers - 1):
                neurons = nNeuronsLast
            neurons = int(nNeuronsFirst * (taper ** iLayer))
            
            tX = tkl.Dense(neurons, activation=config['activation'], name=f'dense{iLayer}')(tX)
                
        tX = tkl.Dropout(config['dropout'], name='dropout')(tX)
                    
        tOutput = tkl.Dense(1, activation='sigmoid')(tX)
            
        model = tf.keras.Model((tInput, tInputZ), tOutput)
        model.compile(loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                      optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
        return model
    
    def cross_validate(self):
        lsResults = []
        nInnerFolds = self.kfolds.n_folds_inner
        
        for iFold in range(nInnerFolds):
            dfXTrain, dfZTrain, dfYTrain, dfXVal, dfZVal, dfYVal = self.kfolds.get_fold(self.config['outer_fold'],
                                                                                        idx_inner=iFold)
            
            scaler = StandardScaler()
            imputer = SimpleImputer()
            
            arrXTrain = scaler.fit_transform(dfXTrain)
            arrXTrain = imputer.fit_transform(arrXTrain)
            arrXVal = scaler.transform(dfXVal)
            arrXVal = imputer.transform(arrXVal)
            
            tf.random.set_seed(self.config['seed'])
            model: tf.keras.Model = self.create_model(self.config)
            dictClassWeights = {0: dfYTrain.mean(), 1: (1 - dfYTrain.mean())}
            
            log = model.fit((arrXTrain, dfZTrain), dfYTrain,
                            validation_data=((arrXVal, dfZVal), dfYVal),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auprc', patience=10)],
                            epochs=100,
                            verbose=0,
                            class_weight=dictClassWeights)
            nTrainedEpochs = len(log.history['loss']) - 10
            
            arrYPredTrain = model.predict((arrXTrain, dfZTrain))
            dictMetricsTrain, youden = classification_metrics(dfYTrain, arrYPredTrain)
            dictMetricsTrain.update(Partition='Train', Epochs=nTrainedEpochs)
            
            arrYPredVal = model.predict((arrXVal, dfZVal))
            dictMetricsVal, _ = classification_metrics(dfYVal, arrYPredVal, youden)
            dictMetricsVal.update(Partition='Val', Epochs=nTrainedEpochs)
            
            lsResults += [dictMetricsTrain, dictMetricsVal]
            
        dfResults = pd.DataFrame(lsResults)
        dfResults.to_csv(os.path.join(self.logdir, 'inner_crossval.csv'))        
        
        # dfResultsTrain = dfResults.loc[dfResults['Partition'] == 'Train']    
        dfResultsVal = dfResults.loc[dfResults['Partition'] == 'Val']
        
        return dfResultsVal.mean().to_dict()
    
    def final_test(self, epochs):
        dfXTrain, dfZTrain, dfYTrain, dfXVal, dfZVal, dfYVal = self.kfolds.get_fold(self.config['outer_fold'])
        
        scaler = StandardScaler()
        imputer = SimpleImputer()
        
        arrXTrain = scaler.fit_transform(dfXTrain)
        arrXTrain = imputer.fit_transform(arrXTrain)
        arrXVal = scaler.transform(dfXVal)
        arrXVal = imputer.transform(arrXVal)
        
        tf.random.set_seed(self.config['seed'])
        model: tf.keras.Model = self.create_model(self.config)
        dictClassWeights = {0: dfYTrain.mean(), 1: (1 - dfYTrain.mean())}
        
        log = model.fit((arrXTrain, dfZTrain), dfYTrain,
                        validation_data=((arrXVal, dfZVal), dfYVal),
                        epochs=epochs,
                        verbose=0,
                        class_weight=dictClassWeights)
        model.save(self.logdir)
        
        arrYPredTrain = model.predict((arrXTrain, dfZTrain))
        dictMetricsTrain, youden = classification_metrics(dfYTrain, arrYPredTrain)
        dictMetricsTrain.update(Partition='Train')
        
        arrYPredVal = model.predict((arrXVal, dfZVal))
        dictMetricsVal, _ = classification_metrics(dfYVal, arrYPredVal, youden)
        dictMetricsVal.update(Partition='Val')
            
        dfResults = pd.DataFrame([dictMetricsTrain, dictMetricsVal])
        dfResults.to_csv(os.path.join(self.logdir, 'final_test.csv'))        
        
        # dfResultsTrain = dfResults.loc[dfResults['Partition'] == 'Train']    
        dfResultsVal = dfResults.loc[dfResults['Partition'] == 'Val']
        
        print(dfResultsVal.mean())
        return


class MetaLearningModel(BaseModel):
    
    def cross_validate(self):
        from medl.models.metalearning import mldg
        
        lsResults = []
        nInnerFolds = self.kfolds.n_folds_inner
        
        for iFold in range(nInnerFolds):
            dfXTrain, dfZTrain, dfYTrain, dfXVal, dfZVal, dfYVal = self.kfolds.get_fold(self.config['outer_fold'],
                                                                                        idx_inner=iFold)
            
            scaler = StandardScaler()
            imputer = SimpleImputer()
            
            arrXTrain = scaler.fit_transform(dfXTrain)
            arrXTrain = imputer.fit_transform(arrXTrain)
            arrXVal = scaler.transform(dfXVal)
            arrXVal = imputer.transform(arrXVal)
            
            tf.random.set_seed(self.config['seed'])
            model: tf.keras.Model = self.create_model(self.config)
            dictClassWeights = {0: dfYTrain.mean(), 1: (1 - dfYTrain.mean())}
            
            mldg(arrXTrain, dfYTrain, dfZTrain.values,
                 model, outer_lr=self.config['learning_rate'],
                 epochs=10,
                 loss_fn=tf.keras.losses.binary_crossentropy)
                        
            arrYPredTrain = model.predict((arrXTrain, dfZTrain))
            dictMetricsTrain, youden = classification_metrics(dfYTrain, arrYPredTrain)
            dictMetricsTrain.update(Partition='Train', Epochs=10)
            
            arrYPredVal = model.predict((arrXVal, dfZVal))
            dictMetricsVal, _ = classification_metrics(dfYVal, arrYPredVal, youden)
            dictMetricsVal.update(Partition='Val', Epochs=10)
            
            lsResults += [dictMetricsTrain, dictMetricsVal]
            
        dfResults = pd.DataFrame(lsResults)
        dfResults.to_csv(os.path.join(self.logdir, 'inner_crossval.csv'))        
        
        # dfResultsTrain = dfResults.loc[dfResults['Partition'] == 'Train']    
        dfResultsVal = dfResults.loc[dfResults['Partition'] == 'Val']
        
        return dfResultsVal.mean().to_dict()
    
    def final_test(self, epochs):
        from medl.models.metalearning import mldg
        
        dfXTrain, dfZTrain, dfYTrain, dfXVal, dfZVal, dfYVal = self.kfolds.get_fold(self.config['outer_fold'])
        
        scaler = StandardScaler()
        imputer = SimpleImputer()
        
        arrXTrain = scaler.fit_transform(dfXTrain)
        arrXTrain = imputer.fit_transform(arrXTrain)
        arrXVal = scaler.transform(dfXVal)
        arrXVal = imputer.transform(arrXVal)
        
        tf.random.set_seed(self.config['seed'])
        model: tf.keras.Model = self.create_model(self.config)
        dictClassWeights = {0: dfYTrain.mean(), 1: (1 - dfYTrain.mean())}
        
        mldg(arrXTrain, dfYTrain, dfZTrain.values,
            model, outer_lr=self.config['learning_rate'],
            epochs=epochs,
            loss_fn=tf.keras.losses.binary_crossentropy,
            verbose=True)
        model.save(self.logdir)
        
        arrYPredTrain = model.predict((arrXTrain, dfZTrain))
        dictMetricsTrain, youden = classification_metrics(dfYTrain, arrYPredTrain)
        dictMetricsTrain.update(Partition='Train')
        
        arrYPredVal = model.predict((arrXVal, dfZVal))
        dictMetricsVal, _ = classification_metrics(dfYVal, arrYPredVal, youden)
        dictMetricsVal.update(Partition='Val')
            
        dfResults = pd.DataFrame([dictMetricsTrain, dictMetricsVal])
        dfResults.to_csv(os.path.join(self.logdir, 'final_test.csv'))        
        
        # dfResultsTrain = dfResults.loc[dfResults['Partition'] == 'Train']    
        dfResultsVal = dfResults.loc[dfResults['Partition'] == 'Val']
        
        print(dfResultsVal.mean())
        return


class MixedEffectsModel(SiteInputModel):
    
    # def create_model(self, config) -> tf.keras.Model:
    #     """Generates a MLP model

    #     Args:
    #         config (dict): hyperparameters

    #     Returns:
    #         tf.keras.Model: model
    #     """
    #     tf.get_logger().setLevel('ERROR')
                
    #     tInput = tkl.Input(self.kfolds.x.shape[1], name='input_x')
    #     tInputZ = tkl.Input(self.kfolds.z.shape[1], name='input_z')
    #     tX = tInput
        
    #     if config['re_type'] == 'linear_slope':
    #         # RE slope multiplied by inputs before nonlinear transformations
    #         tRE = RandomEffects(units=self.kfolds.x.shape[1],
    #                             prior_scale=config['prior_scale'],
    #                             kl_weight=config['kl_weight'],
    #                             l1_weight=config['l1_weight'],
    #                             name='re_slopes')(tInputZ)
    #         tX = tkl.Concatenate(axis=-1)([tX, tX * tRE])
        
    #     # Round from float to int (since SkOpt doesn't sample integers)
    #     nNeuronsFirst = int(config['layer_1_neurons'])
    #     nNeuronsLast = int(config['last_layer_neurons'])
    #     nLayers = int(config['hidden_layers'])
        
    #     taper = np.power(nNeuronsLast / nNeuronsFirst, 1 / nLayers)
        
    #     for iLayer in range(nLayers):
    #         if iLayer == (nLayers - 1):
    #             neurons = nNeuronsLast
    #         neurons = int(nNeuronsFirst * (taper ** iLayer))
            
    #         tX = tkl.Dense(neurons, activation=config['activation'], name=f'dense{iLayer}')(tX)
        
    #     if config['re_type'] == 'nonlinear_slope':
    #         # RE slope multiplied by inputs after nonlinear transformations
    #         tRE = RandomEffects(units=neurons,
    #                             prior_scale=config['prior_scale'],
    #                             kl_weight=config['kl_weight'],
    #                             l1_weight=config['l1_weight'],
    #                             name='re_slopes')(tInputZ)
    #         tX = tkl.Concatenate(axis=-1)([tX, tX * tRE])
        
    #     tX = tkl.Dropout(config['dropout'], name='dropout')(tX)
           
    #     if config['re_intercept']:
    #         tREIntercept = RandomEffects(units=1,
    #                                      prior_scale=config['prior_scale'],
    #                                      kl_weight=config['kl_weight'],
    #                                      l1_weight=config['l1_weight'],
    #                                      name='re_intercepts')(tInputZ)
    #         tX = tkl.Concatenate(axis=-1)([tX, tREIntercept])
                    
    #     tOutput = tkl.Dense(1, activation='sigmoid')(tX)
            
    #     model = tf.keras.Model((tInput, tInputZ), tOutput)
    #     model.compile(loss='binary_crossentropy',
    #                     metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
    #                     optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
    #     return model
    
    def create_model(self, config) -> tf.keras.Model:
        """Generates a MLP model

        Args:
            config (dict): hyperparameters

        Returns:
            tf.keras.Model: model
        """
        tf.get_logger().setLevel('ERROR')
                
        tInput = tkl.Input(self.kfolds.x.shape[1], name='input_x')
        tInputZ = tkl.Input(self.kfolds.z.shape[1], name='input_z')
        tX = tInput
        
        # RE slope multiplied by inputs before nonlinear transformations
        tRE = RandomEffects(units=self.kfolds.x.shape[1],
                            prior_scale=config['prior_scale'],
                            kl_weight=config['kl_weight'],
                            l1_weight=config['l1_weight'],
                            name='re_slopes')(tInputZ)
        
        if config['re_type'] == 'first_layer':
            tX = tkl.Concatenate(axis=-1)([tX, tInput * tRE])
        
        # Round from float to int (since SkOpt doesn't sample integers)
        nNeuronsFirst = int(config['layer_1_neurons'])
        nNeuronsLast = int(config['last_layer_neurons'])
        nLayers = int(config['hidden_layers'])
        
        taper = np.power(nNeuronsLast / nNeuronsFirst, 1 / nLayers)
        
        for iLayer in range(nLayers):
            if iLayer == (nLayers - 1):
                neurons = nNeuronsLast
            neurons = int(nNeuronsFirst * (taper ** iLayer))
            
            tX = tkl.Dense(neurons, activation=config['activation'], name=f'dense{iLayer}')(tX)
                
        tX = tkl.Dropout(config['dropout'], name='dropout')(tX)
        
        if config['re_type'] == 'last_layer':
            tX = tkl.Concatenate(axis=-1)([tX, tInput * tRE])
           
        if config['re_intercept']:
            tREIntercept = RandomEffects(units=1,
                                         prior_scale=config['prior_scale'],
                                         kl_weight=config['kl_weight'],
                                         l1_weight=config['l1_weight'],
                                         name='re_intercepts')(tInputZ)
            tX = tkl.Concatenate(axis=-1)([tX, tREIntercept])
                    
        tOutput = tkl.Dense(1, activation='sigmoid')(tX)
            
        model = tf.keras.Model((tInput, tInputZ), tOutput)
        model.compile(loss='binary_crossentropy',
                        metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                        optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
        return model
