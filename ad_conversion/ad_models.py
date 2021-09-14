import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow_probability.python.layers as tpl
import tensorflow_probability.python.distributions.normal as normal_lib
import tensorflow_probability.python.distributions.independent as independent_lib
import tensorflow_probability.python.distributions.kullback_leibler as kl_lib
from medl.models.random_effects import RandomEffects
from medl.crossvalidation.splitting import NestedKFoldUtil
from medl.metrics import classification_metrics

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

'''
Basic MLP binary classifiers with 2 hidden layers and single output. 
'''

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
        """Model wrapper for use with Tune trials. Creates a simple MLP binary classifier
        according to a provided dictionary of hyperparameters.
        
        Hyperparameters include:
        * folds: path to pickled NestedKFoldUtil object
        * outer_fold: index of outer CV fold
        * layer_1_neurons: number of neurons in 1st hidden layer
        * last_layer_neurons: number of neurons in last hidden layer
        * hidden_layers: number of hidden layers 
        * activation: activation function name
        * dropout: dropout rate before output layer
        * learning rate: learning rate for Adam optimizer
        
        Hidden layers linearly taper in size between the 1st and last layers.

        Args:
            config (dict): hyperparameters provided by Tune search algorithm
            logdir (str): output path
        """        
                
        self.config = config
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        
        with open(config['folds'], 'rb') as f:
            self.kfolds: NestedKFoldUtil = pickle.load(f) 
                
    def create_model(self, config) -> tf.keras.Model: 
        """Generates an MLP model.

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
        """Perform cross-validation and return the mean performance across inner
        validation folds. Also saves the complete per-fold training and
        validation performance as a CSV file inside the logdir.
        
        Models are trained for 100 epochs with early-stopping. 

        Returns: dict: mean inner validation performance, including AUROC, bal. 
        acc., etc.
        """        
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
    
    def _save_model(self, model, path):
        model.save(path)
    
    def final_test(self, epochs):
        """Train model and evaluate on the final test set (outer validation). 

        Args:
            epochs (int): training duration
            
        Returns: dict: test performance, including AUROC, bal. acc., etc.
        """        
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
        self._save_model(model, self.logdir)
        
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
        """Generates an MLP with an additional input for site membership. This
        is concatenated to the main input before the first hidden layer.

        Args: 
            config (dict): hyperparameters

        Returns: tf.keras.Model: model
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
        """Perform cross-validation and return the mean performance across inner
        validation folds. Also saves the complete per-fold training and
        validation performance as a CSV file inside the logdir.
        
        Models are trained for 100 epochs with early-stopping. 

        Returns: dict: mean inner validation performance, including AUROC, bal. 
        acc., etc.
        """  
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
        """Train model and evaluate on the final test set (outer validation). 

        Args:
            epochs (int): training duration
            
        Returns: dict: test performance, including AUROC, bal. acc., etc.
        """  
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
        self._save_model(model, self.logdir)
        
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
    
    def cross_validate(self, epochs):
        """Perform cross-validation and return the mean performance across inner
        validation folds. Also saves the complete per-fold training and
        validation performance as a CSV file inside the logdir.
        
        Models are trained using the Meta-Learning domain generalization technique 
        from Li et al. 2018.
        
        Args:
            epochs (int): training duration

        Returns: dict: mean inner validation performance, including AUROC, bal. 
        acc., etc.
        """  
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
                 epochs=epochs,
                 loss_fn=tf.keras.losses.binary_crossentropy)
                        
            arrYPredTrain = model.predict((arrXTrain, dfZTrain))
            dictMetricsTrain, youden = classification_metrics(dfYTrain, arrYPredTrain)
            dictMetricsTrain.update(Partition='Train', Epochs=epochs)
            
            arrYPredVal = model.predict((arrXVal, dfZVal))
            dictMetricsVal, _ = classification_metrics(dfYVal, arrYPredVal, youden)
            dictMetricsVal.update(Partition='Val', Epochs=epochs)
            
            lsResults += [dictMetricsTrain, dictMetricsVal]
            
        dfResults = pd.DataFrame(lsResults)
        dfResults.to_csv(os.path.join(self.logdir, 'inner_crossval.csv'))        
        
        # dfResultsTrain = dfResults.loc[dfResults['Partition'] == 'Train']    
        dfResultsVal = dfResults.loc[dfResults['Partition'] == 'Val']
        
        return dfResultsVal.mean().to_dict()
    
    def final_test(self, epochs):
        """Train model using meta-learning domain generalization and evaluate 
        on the final test set (outer validation). 

        Args:
            epochs (int): training duration
            
        Returns: dict: test performance, including AUROC, bal. acc., etc.
        """  
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
        self._save_model(model, self.logdir)
        
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
    def __init__(self, config, logdir):
        """Model wrapper for use with Tune trials. Creates a mixed effects MLP
        binary classifier according to a provided dictionary of hyperparameters.

        Hyperparameters include:
        * folds: path to pickled NestedKFoldUtil object
        * outer_fold: index of outer CV fold
        * layer_1_neurons: number of neurons in 1st hidden layer
        * last_layer_neurons: number of neurons in last hidden layer
        * hidden_layers: number of hidden layers 
        * activation: activation function name
        * dropout: dropout rate before output layer
        * learning rate: learning rate for Adam optimizer
        * re_prior_scale: s.d. of random effects prior distributions
        * re_kl_weight: weight of KL divergence loss
        * re_l1_weight: weight of L1 regularization on random effect posterior
          distributions
        * re_type: 'first_layer' or 'last_layer'; where to join the random slope
          layer into the model, either before the first hidden layer or after the
          last hidden layer. 
        * re_intercept: boolean; whether to include a random intercept after the
          last hidden layer

        Hidden layers linearly taper in size between the 1st and last layers.

        Args: 
            config (dict): hyperparameters provided by Tune search algorithm
            logdir (str): output path
        """              
        super().__init__(config, logdir)
    
    def create_model(self, config) -> tf.keras.Model:
        """Generates a mixed effects MLP model.

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
                            prior_scale=config['re_prior_scale'],
                            kl_weight=config['re_kl_weight'],
                            l1_weight=config['re_l1_weight'],
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
                                         prior_scale=config['re_prior_scale'],
                                         kl_weight=config['re_kl_weight'],
                                         l1_weight=config['re_l1_weight'],
                                         name='re_intercepts')(tInputZ)
            tX = tkl.Concatenate(axis=-1)([tX, tREIntercept])
                    
        tOutput = tkl.Dense(1, activation='sigmoid')(tX)
            
        model = tf.keras.Model((tInput, tInputZ), tOutput)
        model.compile(loss='binary_crossentropy',
                        metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                        optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
        return model


class BaseModelBayesian(BaseModel):
    
    @staticmethod
    def _make_prior_fn(config):
        
        def prior_fn(dtype, shape, name, trainable, add_variable_fn):
            del name, trainable, add_variable_fn
            dist = normal_lib.Normal(loc=tf.zeros(shape, dtype),
                                     scale=dtype.as_numpy_dtype(config['prior_scale']))
            batch_ndims = tf.size(dist.batch_shape_tensor())
            return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
        
        return prior_fn
    
    def create_model(self, config) -> tf.keras.Model:
        """Generates an MLP model using Bayesian dense layers

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
        
        divergence_fn = lambda q, p, ignore: config['kl_weight'] * kl_lib.kl_divergence(q, p)
        
        for iLayer in range(nLayers):
            if iLayer == (nLayers - 1):
                neurons = nNeuronsLast
            neurons = int(nNeuronsFirst * (taper ** iLayer))
            
            tX = tpl.DenseFlipout(neurons, 
                                  activation=config['activation'], 
                                  kernel_prior_fn=self._make_prior_fn(config),
                                  kernel_divergence_fn=divergence_fn,
                                  bias_prior_fn=self._make_prior_fn(config),
                                  bias_divergence_fn=divergence_fn,
                                  name=f'dense{iLayer}')(tX)
                    
        tX = tkl.Dropout(config['dropout'], name='dropout')(tX)
                    
        tOutput = tpl.DenseFlipout(1, activation='sigmoid', name='output')(tX)
            
        model = tf.keras.Model(tInput, tOutput)
        model.compile(loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                      optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
        return model
    
    def _save_model(self, model, path):
        # Keras Model.save() doesn't work with TFP layers, so need to save only the weights
        model.save_weights(os.path.join(path, 'model_weights.h5'))


class SiteInputModelBayesian(SiteInputModel, BaseModelBayesian):
    
    def create_model(self, config) -> tf.keras.Model:
        """Generates an MLP model with additional site membership input 
        using Bayesian dense layers

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
        
        divergence_fn = lambda q, p, ignore: config['kl_weight'] * kl_lib.kl_divergence(q, p)
        
        for iLayer in range(nLayers):
            if iLayer == (nLayers - 1):
                neurons = nNeuronsLast
            neurons = int(nNeuronsFirst * (taper ** iLayer))
            
            tX = tpl.DenseFlipout(neurons, 
                                  activation=config['activation'], 
                                  kernel_prior_fn=self._make_prior_fn(config),
                                  kernel_divergence_fn=divergence_fn,
                                  bias_prior_fn=self._make_prior_fn(config),
                                  bias_divergence_fn=divergence_fn,
                                  name=f'dense{iLayer}')(tX)
                    
        tX = tkl.Dropout(config['dropout'], name='dropout')(tX)
                    
        tOutput = tpl.DenseFlipout(1, activation='sigmoid', name='output')(tX)
            
        model = tf.keras.Model((tInput, tInputZ), tOutput)
        model.compile(loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                      optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
        return model
    
    def _save_model(self, model, path):
        BaseModelBayesian._save_model(self, model, path)
        

class MetaLearningModelBayesian(MetaLearningModel, BaseModelBayesian):
    def __init__(self, config, logdir):
        '''
        Bayesian MLP with training via meta-learning domain generalization.
        '''
        super().__init__(config, logdir)
    
    # Inherit Bayesian model creation and saving methods
    def create_model(self, config) -> tf.keras.Model:
        return BaseModelBayesian.create_model(self, config)

    def _save_model(self, model, path):
        BaseModelBayesian._save_model(self, model, path)

class MixedEffectsModelBayesian(SiteInputModel):
    
    @staticmethod
    def _make_prior_fn(config):
        
        def prior_fn(dtype, shape, name, trainable, add_variable_fn):
            del name, trainable, add_variable_fn
            dist = normal_lib.Normal(loc=tf.zeros(shape, dtype),
                                     scale=dtype.as_numpy_dtype(config['fe_prior_scale']))
            batch_ndims = tf.size(dist.batch_shape_tensor())
            return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
        
        return prior_fn
    
    def create_model(self, config) -> tf.keras.Model:
        """Generates a mixed effects MLP model with Bayesian dense layers.

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
                            prior_scale=config['re_prior_scale'],
                            kl_weight=config['re_kl_weight'],
                            l1_weight=config['re_l1_weight'],
                            name='re_slopes')(tInputZ)
        
        if config['re_type'] == 'first_layer':
            tX = tkl.Concatenate(axis=-1)([tX, tInput * tRE])
        
        # Round from float to int (since SkOpt doesn't sample integers)
        nNeuronsFirst = int(config['layer_1_neurons'])
        nNeuronsLast = int(config['last_layer_neurons'])
        nLayers = int(config['hidden_layers'])
        
        taper = np.power(nNeuronsLast / nNeuronsFirst, 1 / nLayers)
        
        divergence_fn = lambda q, p, ignore: config['fe_kl_weight'] * kl_lib.kl_divergence(q, p)
        
        for iLayer in range(nLayers):
            if iLayer == (nLayers - 1):
                neurons = nNeuronsLast
            neurons = int(nNeuronsFirst * (taper ** iLayer))
            
            tX = tpl.DenseFlipout(neurons, 
                                  activation=config['activation'], 
                                  kernel_prior_fn=self._make_prior_fn(config),
                                  kernel_divergence_fn=divergence_fn,
                                  bias_prior_fn=self._make_prior_fn(config),
                                  bias_divergence_fn=divergence_fn,
                                  name=f'dense{iLayer}')(tX)
                
        tX = tkl.Dropout(config['dropout'], name='dropout')(tX)
        
        if config['re_type'] == 'last_layer':
            tX = tkl.Concatenate(axis=-1)([tX, tInput * tRE])
           
        if config['re_intercept']:
            tREIntercept = RandomEffects(units=1,
                                         prior_scale=config['prior_scale'],
                                         kl_weight=config['re_kl_weight'],
                                         l1_weight=config['re_l1_weight'],
                                         name='re_intercepts')(tInputZ)
            tX = tkl.Concatenate(axis=-1)([tX, tREIntercept])
                    
        tOutput = tpl.DenseFlipout(1, 
                                   kernel_prior_fn=self._make_prior_fn(config),
                                   kernel_divergence_fn=divergence_fn,
                                   bias_prior_fn=self._make_prior_fn(config),
                                   bias_divergence_fn=divergence_fn,
                                   activation='sigmoid', name='output')(tX)
            
        model = tf.keras.Model((tInput, tInputZ), tOutput)
        model.compile(loss='binary_crossentropy',
                        metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')],
                        optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']))
        
        return model
    
    def _save_model(self, model, path):
        # Keras Model.save() doesn't work with TFP layers, so need to save only the weights
        model.save_weights(os.path.join(path, 'model_weights.h5'))