import tensorflow as tf
import tensorflow.keras.layers as tkl

import tensorflow_probability.python.layers as tpl
import tensorflow_probability.python.distributions as tpd

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut

from ..metrics import classification_metrics
from .random_effects import GammaRandomEffects, RandomEffects

def adversarial_autoencoder(n_latent_dims=56, input_shape=(256, 256, 1), recon_loss_weight=0.99, adv_loss_weight=0.01):
    encoder_in = tkl.Input(input_shape, name='e_input')
    # Encoder
    x = tkl.Conv2D(64, 4, strides=(2, 2), padding='same', name='e_conv_0')(encoder_in)
    x = tkl.BatchNormalization(name='e_batchnorm_0')(x)
    x = tkl.PReLU(name='e_prelu_0')(x)

    x = tkl.Conv2D(128, 4, strides=(2, 2), padding='same', name='e_conv_1')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_1')(x)
    x = tkl.PReLU(name='e_prelu_1')(x)

    x = tkl.Conv2D(256, 4, strides=(2, 2), padding='same', name='e_conv_2')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_2')(x)
    x = tkl.PReLU(name='e_prelu_2')(x)

    x = tkl.Conv2D(512, 4, strides=(2, 2), padding='same', name='e_conv_3')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_3')(x)
    x = tkl.PReLU(name='e_prelu_3')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_4')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_4')(x)
    x = tkl.PReLU(name='e_prelu_4')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_5')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_5')(x)
    x = tkl.PReLU(name='e_prelu_5')(x)

    # Latent representation
    x = tkl.Flatten(name='flatten')(x)
    latent = tkl.Dense(n_latent_dims, name='latent')(x)
    encoder_out = tkl.BatchNormalization(name='e_output')(latent)

    # Decoder
    decoder_in = tkl.Input(n_latent_dims, name='d_input')
    x = tkl.Dense(1024 * input_shape[0] // 64 * input_shape[1] // 64, name='d_dense')(decoder_in)
    x = tkl.Reshape((input_shape[0] // 64, input_shape[1] // 64, 1024), name='reshape')(x)
    x = tkl.PReLU(name='d_prelu_dense')(x)

    x = tkl.Conv2DTranspose(1024, 4, strides=(2, 2), padding='same', name='d_deconv_0')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_0')(x)
    x = tkl.PReLU(name='d_prelu_0')(x)

    x = tkl.Conv2DTranspose(512, 4, strides=(2, 2), padding='same', name='d_deconv_1')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_1')(x)
    x = tkl.PReLU(name='d_prelu_1')(x)

    x = tkl.Conv2DTranspose(256, 4, strides=(2, 2), padding='same', name='d_deconv_2')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_2')(x)
    x = tkl.PReLU(name='d_prelu_2')(x)

    x = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='d_deconv_3')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_3')(x)
    x = tkl.PReLU(name='d_prelu_3')(x)

    x = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='d_deconv_4')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_4')(x)
    x = tkl.PReLU(name='d_prelu_4')(x)

    x = tkl.Conv2DTranspose(1, 4, strides=(2, 2), padding='same', name='d_deconv_5')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_5')(x)
    decoder_out = tkl.Activation('sigmoid', name='d_output')(x)

    # Adversary
    adversary_in = tkl.Input(n_latent_dims, name='a_input')
    x = tkl.Dense(1024, name='a_dense_0')(adversary_in)
    x = tkl.LeakyReLU(name='a_leakyrelu_0')(x)
    
    x = tkl.Dense(1024, name='a_dense_1')(x)
    x = tkl.BatchNormalization(name='a_batchnorm_1')(x)
    x = tkl.LeakyReLU(name='a_leakyrelu_1')(x)

    x = tkl.Dense(512, name='a_dense_2')(x)
    x = tkl.BatchNormalization(name='a_batchnorm_2')(x)
    x = tkl.LeakyReLU(name='a_leakyrelu_2')(x)

    adversary_out = tkl.Dense(1, activation='sigmoid')(x)

    # Put these sub-models together
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    encoder = tf.keras.models.Model(encoder_in, encoder_out)
    decoder = tf.keras.models.Model(decoder_in, decoder_out)

    adversary = tf.keras.models.Model(adversary_in, adversary_out)
    adversary.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Make the adversary's layers untrainable when it's part of the autoencoder
    # It is still trainable in the original adversary model
    adversary.trainable = False

    autoencoder_in = tkl.Input(input_shape, name='e_input')
    autoencoder_latent = encoder(autoencoder_in)
    autoencoder_out = decoder(autoencoder_latent)
    validity = adversary(autoencoder_latent)
    autoencoder = tf.keras.models.Model(autoencoder_in, [autoencoder_out, validity])
    autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[recon_loss_weight, adv_loss_weight], optimizer=opt,
                        metrics=[['mean_squared_error'], ['accuracy']])
    
    return autoencoder, adversary, encoder


def test_classification(latents: np.array, image_list: pd.DataFrame):
    """Train LDA classifier and return the leave-one-PDX-out test AUROC.

    Args:
        latents (np.array): latent embeddings
        image_list (pd.DataFrame): dataframe with columns [cell, celltype, met-eff, date]

    Returns:
        [type]: [description]
    """    
    
    dfEmbeddings = pd.DataFrame(latents, columns=[f'latent{x}' for x in range(latents.shape[1])])
    dfEmbeddings['Cell'] = image_list['cell']

    dfCellInfo = image_list.loc[~image_list['cell'].duplicated(keep='first')]
    dfCellInfo.index = dfCellInfo['cell']

    # Compute time average latent for each cell
    dfData = dfEmbeddings.groupby('Cell').mean()
    dfData['Label'] = dfCellInfo['met-eff'].loc[dfData.index]
    dfData['PDX'] = dfCellInfo['celltype'].loc[dfData.index]
    dfData['Date'] = dfCellInfo['date'].loc[dfData.index]
    dfData['PDX_Date'] = dfData['PDX'].combine(dfData['Date'], lambda pdx, date: pdx + '_' + str(date))
    
    dfLatents = dfData.filter(like='latent', axis=1)
    dfLabels = dfData[['Label', 'PDX', 'Date', 'PDX_Date']].convert_dtypes()
 
    # Normalize latents to 0M1V
    dfLatentsNorm = dfLatents - dfLatents.mean(axis=0)
    dfLatentsNorm = dfLatentsNorm / dfLatents.std(axis=0)

    # 1 for low, 0 for high
    dfLabels['Low met. eff.'] = (dfLabels['Label'] == 'low').astype(int)

    splitter = LeaveOneGroupOut()
    lsPredictionsRaw = []
    for arrTrainIdx, arrTestIdx in splitter.split(dfLatentsNorm, dfLabels['Low met. eff.'], groups=dfLabels['PDX']):
        
        dfLatentsTrain = dfLatentsNorm.iloc[arrTrainIdx]
        dfLabelsTrain = dfLabels.iloc[arrTrainIdx]
        dfLatentsTest = dfLatentsNorm.iloc[arrTestIdx]
        dfLabelsTest = dfLabels.iloc[arrTestIdx]
        
        # Remove any training samples that come from the same PDX
        dfLabelsTrain = dfLabelsTrain.loc[dfLabelsTrain['PDX'] != dfLabelsTest['PDX'].iloc[0]]
        dfLatentsTrain = dfLatentsTrain.loc[dfLabelsTrain.index]

        classifier = LinearDiscriminantAnalysis()
        # transform returns the linear projection, while predict_proba returns the probabilistic predictions
        arrPredRawTrain = classifier.fit_transform(dfLatentsTrain, dfLabelsTrain['Low met. eff.'])
        
        # Unthresholded predictions
        arrPredRawTest = np.zeros((dfLabelsTest.shape[0], 2))
        arrPredRawTest[:, 0] = dfLabelsTest['Low met. eff.']
        arrPredRawTest[:, 1] = classifier.transform(dfLatentsTest).squeeze()
        lsPredictionsRaw += [arrPredRawTest]
    
    # Compute AUROC using the unthresholded predictions on test data
    arrPredRawPerCell = np.concatenate(lsPredictionsRaw)
    dictScoresTest, _ = classification_metrics(arrPredRawPerCell[:, 0], arrPredRawPerCell[:, 1])
    auroc = dictScoresTest['AUROC']
    return auroc


def autoencoder_classifier(n_latent_dims=56, input_shape=(256, 256, 1), recon_loss_weight=1.0, class_loss_weight=0.01):
    encoder_in = tkl.Input(input_shape, name='e_input')
    # Encoder
    x = tkl.Conv2D(64, 4, strides=(2, 2), padding='same', name='e_conv_0')(encoder_in)
    x = tkl.BatchNormalization(name='e_batchnorm_0')(x)
    x = tkl.PReLU(name='e_prelu_0')(x)

    x = tkl.Conv2D(128, 4, strides=(2, 2), padding='same', name='e_conv_1')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_1')(x)
    x = tkl.PReLU(name='e_prelu_1')(x)

    x = tkl.Conv2D(256, 4, strides=(2, 2), padding='same', name='e_conv_2')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_2')(x)
    x = tkl.PReLU(name='e_prelu_2')(x)

    x = tkl.Conv2D(512, 4, strides=(2, 2), padding='same', name='e_conv_3')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_3')(x)
    x = tkl.PReLU(name='e_prelu_3')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_4')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_4')(x)
    x = tkl.PReLU(name='e_prelu_4')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_5')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_5')(x)
    x = tkl.PReLU(name='e_prelu_5')(x)

    # Latent representation
    x = tkl.Flatten(name='flatten')(x)
    latent = tkl.Dense(n_latent_dims, name='latent')(x)
    encoder_out = tkl.BatchNormalization(name='e_output')(latent)

    # Decoder
    decoder_in = tkl.Input(n_latent_dims, name='d_input')
    x = tkl.Dense(1024 * input_shape[0] // 64 * input_shape[1] // 64, name='d_dense')(decoder_in)
    x = tkl.Reshape((input_shape[0] // 64, input_shape[1] // 64, 1024), name='reshape')(x)
    x = tkl.PReLU(name='d_prelu_dense')(x)

    x = tkl.Conv2DTranspose(1024, 4, strides=(2, 2), padding='same', name='d_deconv_0')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_0')(x)
    x = tkl.PReLU(name='d_prelu_0')(x)

    x = tkl.Conv2DTranspose(512, 4, strides=(2, 2), padding='same', name='d_deconv_1')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_1')(x)
    x = tkl.PReLU(name='d_prelu_1')(x)

    x = tkl.Conv2DTranspose(256, 4, strides=(2, 2), padding='same', name='d_deconv_2')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_2')(x)
    x = tkl.PReLU(name='d_prelu_2')(x)

    x = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='d_deconv_3')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_3')(x)
    x = tkl.PReLU(name='d_prelu_3')(x)

    x = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='d_deconv_4')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_4')(x)
    x = tkl.PReLU(name='d_prelu_4')(x)

    x = tkl.Conv2DTranspose(1, 4, strides=(2, 2), padding='same', name='d_deconv_5')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_5')(x)
    decoder_out = tkl.Activation('sigmoid', name='d_output')(x)

    # Auxiliary classifier
    classifier_in = tkl.Input(n_latent_dims, name='c_input')    
    x = tkl.Dense(32, name='c_dense_0')(classifier_in)
    x = tkl.LeakyReLU(name='c_leakyrelu_0')(x)
    classifier_out = tkl.Dense(1, activation='sigmoid')(x)

    # Put these sub-models together
    encoder = tf.keras.models.Model(encoder_in, encoder_out, name='encoder')
    decoder = tf.keras.models.Model(decoder_in, decoder_out, name='decoder')
    classifier = tf.keras.models.Model(classifier_in, classifier_out, name='classifier')

    # Combined model
    autoencoder_in = tkl.Input(input_shape, name='autoencoder_input')
    autoencoder_latent = encoder(autoencoder_in)
    autoencoder_out = decoder(autoencoder_latent)
    classification = classifier(autoencoder_latent)
    autoencoder = tf.keras.models.Model(autoencoder_in, [autoencoder_out, classification], name='autoencoder')
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    autoencoder.compile(loss=['mse', 'binary_crossentropy'], 
                        optimizer=opt,
                        loss_weights=[recon_loss_weight, class_loss_weight],
                        metrics=[['mean_squared_error'], [tf.keras.metrics.AUC()]])
    
    return autoencoder


def me_autoencoder_classifier(n_latent_dims=56, n_clusters=10, input_shape=(256, 256, 1), n_re_filters=16, recon_loss_weight=1.0, class_loss_weight=0.01, kl_weight=0.001):
    encoder_in_x = tkl.Input(input_shape, name='e_input_x')
    encoder_in_z = tkl.Input(n_clusters, name='e_input_z')
    
    # Encoder
    conv_re = tkl.Conv2D(n_re_filters, 4, strides=(2, 2), padding='same', name='e_conv_re')(encoder_in_x)
    re = GammaRandomEffects(units=n_re_filters, kl_weight=kl_weight, name='e_re_mult')(encoder_in_z)
    re = tkl.Reshape((1, 1, n_re_filters), name='e_re_reshape')(re)
    conv_prod = re * conv_re
    
    x = tkl.Conv2D(64, 4, strides=(2, 2), padding='same', name='e_conv_0')(encoder_in_x)
    x = tkl.Concatenate(axis=-1, name='e_concat_re')([x, conv_prod])
    x = tkl.BatchNormalization(name='e_batchnorm_0')(x)
    x = tkl.PReLU(name='e_prelu_0')(x)

    x = tkl.Conv2D(128, 4, strides=(2, 2), padding='same', name='e_conv_1')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_1')(x)
    x = tkl.PReLU(name='e_prelu_1')(x)

    x = tkl.Conv2D(256, 4, strides=(2, 2), padding='same', name='e_conv_2')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_2')(x)
    x = tkl.PReLU(name='e_prelu_2')(x)

    x = tkl.Conv2D(512, 4, strides=(2, 2), padding='same', name='e_conv_3')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_3')(x)
    x = tkl.PReLU(name='e_prelu_3')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_4')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_4')(x)
    x = tkl.PReLU(name='e_prelu_4')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_5')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_5')(x)
    x = tkl.PReLU(name='e_prelu_5')(x)

    # Latent representation
    x = tkl.Flatten(name='flatten')(x)
    latent = tkl.Dense(n_latent_dims, name='latent')(x)
    encoder_out = tkl.BatchNormalization(name='e_output')(latent)

    # Decoder
    decoder_in = tkl.Input(n_latent_dims, name='d_input')
    x = tkl.Dense(1024 * input_shape[0] // 64 * input_shape[1] // 64, name='d_dense')(decoder_in)
    x = tkl.Reshape((input_shape[0] // 64, input_shape[1] // 64, 1024), name='d_reshape')(x)
    x = tkl.PReLU(name='d_prelu_dense')(x)

    x = tkl.Conv2DTranspose(1024, 4, strides=(2, 2), padding='same', name='d_deconv_0')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_0')(x)
    x = tkl.PReLU(name='d_prelu_0')(x)

    x = tkl.Conv2DTranspose(512, 4, strides=(2, 2), padding='same', name='d_deconv_1')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_1')(x)
    x = tkl.PReLU(name='d_prelu_1')(x)

    x = tkl.Conv2DTranspose(256, 4, strides=(2, 2), padding='same', name='d_deconv_2')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_2')(x)
    x = tkl.PReLU(name='d_prelu_2')(x)

    x = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='d_deconv_3')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_3')(x)
    x = tkl.PReLU(name='d_prelu_3')(x)

    x = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='d_deconv_4')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_4')(x)
    x = tkl.PReLU(name='d_prelu_4')(x)

    x = tkl.Conv2DTranspose(1, 4, strides=(2, 2), padding='same', name='d_deconv_5')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_5')(x)
    decoder_out = tkl.Activation('sigmoid', name='d_output')(x)

    # Auxiliary classifier
    classifier_in = tkl.Input(n_latent_dims, name='c_input')    
    x = tkl.Dense(32, name='c_dense_0')(classifier_in)
    x = tkl.LeakyReLU(name='c_leakyrelu_0')(x)
    classifier_out = tkl.Dense(1, activation='sigmoid')(x)

    # Put these sub-models together
    encoder = tf.keras.models.Model((encoder_in_x, encoder_in_z), encoder_out, name='encoder')
    decoder = tf.keras.models.Model(decoder_in, decoder_out, name='decoder')
    classifier = tf.keras.models.Model(classifier_in, classifier_out, name='classifier')

    # Combined model
    autoencoder_in_x = tkl.Input(input_shape, name='autoencoder_input_x')
    autoencoder_in_z = tkl.Input(n_clusters, name='autoencoder_input_z')
    autoencoder_latent = encoder((autoencoder_in_x, autoencoder_in_z))
    autoencoder_out = decoder(autoencoder_latent)
    classification = classifier(autoencoder_latent)
    autoencoder = tf.keras.models.Model([autoencoder_in_x, autoencoder_in_z], 
                                        [autoencoder_out, classification], 
                                        name='autoencoder')
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    autoencoder.compile(loss=['mse', 'binary_crossentropy'], 
                        optimizer=opt,
                        loss_weights=[recon_loss_weight, class_loss_weight],
                        metrics=[['mean_squared_error'], [tf.keras.metrics.AUC()]])
    
    return autoencoder


def me_autoencoder_classifier_2layer(n_latent_dims=56, n_clusters=10, 
                                     input_shape=(256, 256, 1), 
                                     n_re_filters_0=16, 
                                     n_re_filters_1=32, 
                                     recon_loss_weight=1.0, 
                                     class_loss_weight=0.01, 
                                     kl_weight=0.001):
    encoder_in_x = tkl.Input(input_shape, name='e_input_x')
    encoder_in_z = tkl.Input(n_clusters, name='e_input_z')
    
    # Encoder
    conv_re_0 = tkl.Conv2D(n_re_filters_0, 4, strides=(2, 2), padding='same', name='e_conv_re_0')(encoder_in_x)
    re_0 = GammaRandomEffects(units=n_re_filters_0, kl_weight=kl_weight, name='e_re_mult_0')(encoder_in_z)
    re_0 = tkl.Reshape((1, 1, n_re_filters_0), name='e_re_reshape_0')(re_0)
    conv_prod_0 = re_0 * conv_re_0
    
    x = tkl.Conv2D(64, 4, strides=(2, 2), padding='same', name='e_conv_0')(encoder_in_x)
    x = tkl.Concatenate(axis=-1, name='e_concat_re')([x, conv_prod_0])
    x = tkl.BatchNormalization(name='e_batchnorm_0')(x)
    x = tkl.PReLU(name='e_prelu_0')(x)

    conv_re_1 = tkl.Conv2D(n_re_filters_1, 4, strides=(2, 2), padding='same', name='e_conv_re_1')(x)
    re_1 = GammaRandomEffects(units=n_re_filters_1, kl_weight=kl_weight, name='e_re_mult_1')(encoder_in_z)
    re_1 = tkl.Reshape((1, 1, n_re_filters_1), name='e_re_reshape_1')(re_1)
    conv_prod_1 = re_1 * conv_re_1

    x = tkl.Conv2D(128, 4, strides=(2, 2), padding='same', name='e_conv_1')(x)
    x = tkl.Concatenate(axis=-1, name='e_concat_re_1')([x, conv_prod_1])
    x = tkl.BatchNormalization(name='e_batchnorm_1')(x)
    x = tkl.PReLU(name='e_prelu_1')(x)

    x = tkl.Conv2D(256, 4, strides=(2, 2), padding='same', name='e_conv_2')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_2')(x)
    x = tkl.PReLU(name='e_prelu_2')(x)

    x = tkl.Conv2D(512, 4, strides=(2, 2), padding='same', name='e_conv_3')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_3')(x)
    x = tkl.PReLU(name='e_prelu_3')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_4')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_4')(x)
    x = tkl.PReLU(name='e_prelu_4')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_5')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_5')(x)
    x = tkl.PReLU(name='e_prelu_5')(x)

    # Latent representation
    x = tkl.Flatten(name='flatten')(x)
    latent = tkl.Dense(n_latent_dims, name='latent')(x)
    encoder_out = tkl.BatchNormalization(name='e_output')(latent)

    # Decoder
    decoder_in = tkl.Input(n_latent_dims, name='d_input')
    x = tkl.Dense(1024 * input_shape[0] // 64 * input_shape[1] // 64, name='d_dense')(decoder_in)
    x = tkl.Reshape((input_shape[0] // 64, input_shape[1] // 64, 1024), name='d_reshape')(x)
    x = tkl.PReLU(name='d_prelu_dense')(x)

    x = tkl.Conv2DTranspose(1024, 4, strides=(2, 2), padding='same', name='d_deconv_0')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_0')(x)
    x = tkl.PReLU(name='d_prelu_0')(x)

    x = tkl.Conv2DTranspose(512, 4, strides=(2, 2), padding='same', name='d_deconv_1')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_1')(x)
    x = tkl.PReLU(name='d_prelu_1')(x)

    x = tkl.Conv2DTranspose(256, 4, strides=(2, 2), padding='same', name='d_deconv_2')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_2')(x)
    x = tkl.PReLU(name='d_prelu_2')(x)

    x = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='d_deconv_3')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_3')(x)
    x = tkl.PReLU(name='d_prelu_3')(x)

    x = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='d_deconv_4')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_4')(x)
    x = tkl.PReLU(name='d_prelu_4')(x)

    x = tkl.Conv2DTranspose(1, 4, strides=(2, 2), padding='same', name='d_deconv_5')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_5')(x)
    decoder_out = tkl.Activation('sigmoid', name='d_output')(x)

    # Auxiliary classifier
    classifier_in = tkl.Input(n_latent_dims, name='c_input')    
    x = tkl.Dense(32, name='c_dense_0')(classifier_in)
    x = tkl.LeakyReLU(name='c_leakyrelu_0')(x)
    classifier_out = tkl.Dense(1, activation='sigmoid')(x)

    # Put these sub-models together
    encoder = tf.keras.models.Model((encoder_in_x, encoder_in_z), encoder_out, name='encoder')
    decoder = tf.keras.models.Model(decoder_in, decoder_out, name='decoder')
    classifier = tf.keras.models.Model(classifier_in, classifier_out, name='classifier')

    # Combined model
    autoencoder_in_x = tkl.Input(input_shape, name='autoencoder_input_x')
    autoencoder_in_z = tkl.Input(n_clusters, name='autoencoder_input_z')
    autoencoder_latent = encoder((autoencoder_in_x, autoencoder_in_z))
    autoencoder_out = decoder(autoencoder_latent)
    classification = classifier(autoencoder_latent)
    autoencoder = tf.keras.models.Model([autoencoder_in_x, autoencoder_in_z], 
                                        [autoencoder_out, classification], 
                                        name='autoencoder')
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    autoencoder.compile(loss=['mse', 'binary_crossentropy'], 
                        optimizer=opt,
                        loss_weights=[recon_loss_weight, class_loss_weight],
                        metrics=[['mean_squared_error'], [tf.keras.metrics.AUC()]])
    
    return autoencoder


def me_autoencoder_classifier_latentscalar(n_latent_dims=56, n_clusters=10,
                                           input_shape=(256, 256, 1), 
                                           recon_loss_weight=1.0, 
                                           class_loss_weight=0.01, 
                                           kl_weight=0.001):
    encoder_in_x = tkl.Input(input_shape, name='e_input_x')
    encoder_in_z = tkl.Input(n_clusters, name='e_input_z')
    
    # Encoder
    x = tkl.Conv2D(64, 4, strides=(2, 2), padding='same', name='e_conv_0')(encoder_in_x)
    x = tkl.BatchNormalization(name='e_batchnorm_0')(x)
    x = tkl.PReLU(name='e_prelu_0')(x)

    x = tkl.Conv2D(128, 4, strides=(2, 2), padding='same', name='e_conv_1')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_1')(x)
    x = tkl.PReLU(name='e_prelu_1')(x)

    x = tkl.Conv2D(256, 4, strides=(2, 2), padding='same', name='e_conv_2')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_2')(x)
    x = tkl.PReLU(name='e_prelu_2')(x)

    x = tkl.Conv2D(512, 4, strides=(2, 2), padding='same', name='e_conv_3')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_3')(x)
    x = tkl.PReLU(name='e_prelu_3')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_4')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_4')(x)
    x = tkl.PReLU(name='e_prelu_4')(x)

    x = tkl.Conv2D(1024, 4, strides=(2, 2), padding='same', name='e_conv_5')(x)
    x = tkl.BatchNormalization(name='e_batchnorm_5')(x)
    x = tkl.PReLU(name='e_prelu_5')(x)

    # Latent representation
    x = tkl.Flatten(name='flatten')(x)
    latent = tkl.Dense(n_latent_dims, name='latent')(x)
    encoder_out = tkl.BatchNormalization(name='e_output')(latent)
    
    reMult = RandomEffects(units=n_latent_dims, name='re_mult')(encoder_in_z)
    encoder_out_me = tkl.Multiply(name='e_output_me')([latent, 1 + reMult])

    # Decoder
    decoder_in = tkl.Input(n_latent_dims, name='d_input')
    x = tkl.Dense(1024 * input_shape[0] // 64 * input_shape[1] // 64, name='d_dense')(decoder_in)
    x = tkl.Reshape((input_shape[0] // 64, input_shape[1] // 64, 1024), name='reshape')(x)
    x = tkl.PReLU(name='d_prelu_dense')(x)

    x = tkl.Conv2DTranspose(1024, 4, strides=(2, 2), padding='same', name='d_deconv_0')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_0')(x)
    x = tkl.PReLU(name='d_prelu_0')(x)

    x = tkl.Conv2DTranspose(512, 4, strides=(2, 2), padding='same', name='d_deconv_1')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_1')(x)
    x = tkl.PReLU(name='d_prelu_1')(x)

    x = tkl.Conv2DTranspose(256, 4, strides=(2, 2), padding='same', name='d_deconv_2')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_2')(x)
    x = tkl.PReLU(name='d_prelu_2')(x)

    x = tkl.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', name='d_deconv_3')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_3')(x)
    x = tkl.PReLU(name='d_prelu_3')(x)

    x = tkl.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', name='d_deconv_4')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_4')(x)
    x = tkl.PReLU(name='d_prelu_4')(x)

    x = tkl.Conv2DTranspose(1, 4, strides=(2, 2), padding='same', name='d_deconv_5')(x)
    x = tkl.BatchNormalization(name='d_batchnorm_5')(x)
    decoder_out = tkl.Activation('sigmoid', name='d_output')(x)

    # Auxiliary classifier
    classifier_in = tkl.Input(n_latent_dims, name='c_input')    
    x = tkl.Dense(32, name='c_dense_0')(classifier_in)
    x = tkl.LeakyReLU(name='c_leakyrelu_0')(x)
    classifier_out = tkl.Dense(1, activation='sigmoid')(x)

    # Put these sub-models together
    encoder = tf.keras.models.Model([encoder_in_x, encoder_in_z], [encoder_out, encoder_out_me], name='encoder')
    decoder = tf.keras.models.Model(decoder_in, decoder_out, name='decoder')
    classifier = tf.keras.models.Model(classifier_in, classifier_out, name='classifier')

    # Combined model
    autoencoder_in_x = tkl.Input(input_shape, name='autoencoder_input_x')
    autoencoder_in_z = tkl.Input(n_clusters, name='autoencoder_input_z')
    
    autoencoder_latent, autoencoder_latent_me = encoder((autoencoder_in_x, autoencoder_in_z))
    autoencoder_out = decoder(autoencoder_latent_me)
    classification = classifier(autoencoder_latent)
    autoencoder = tf.keras.models.Model([autoencoder_in_x, autoencoder_in_z], [autoencoder_out, classification], 
                                        name='autoencoder')
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    autoencoder.compile(loss=['mse', 'binary_crossentropy'], 
                        optimizer=opt,
                        loss_weights=[recon_loss_weight, class_loss_weight],
                        metrics=[['mean_squared_error'], [tf.keras.metrics.AUC()]])
    
    return autoencoder

