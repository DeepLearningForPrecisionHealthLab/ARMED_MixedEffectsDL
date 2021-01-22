'''
Prototype mixed effects adversarial autoencoder
'''
import sys
import tensorflow as tf
import tensorflow.keras.layers as tkl
sys.path.append('../')
from medl.models.random_effects import DenseRandomEffects #pylint: disable=import-error

def me_adversarial_autoencoder(n_groups, 
                            n_latent_dims=56, 
                            input_shape=(256, 256, 1), 
                            recon_loss_weight=0.99, 
                            adv_loss_weight=0.01,
                            random_slope=False,
                            random_int=False):
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
    # Add random effects to this latent rep
    design_in = tkl.Input((n_groups,), name='design_input')
    if random_slope:
        reSlope = DenseRandomEffects(n_latent_dims, loc_init_std=0.1, scale_init_range=(0.01, 0.1), 
                                    prior_scale=0.1,
                                    name='random_slope')(design_in)
        reSlopeMult = tkl.Multiply()([reSlope, decoder_in])
        mixed = tkl.Add()([decoder_in, reSlopeMult])
    else:
        mixed = decoder_in
    if random_int:
        reInt = DenseRandomEffects(1, loc_init_std=0.1, scale_init_range=(0.01, 0.1), 
                               prior_scale=0.1,
                               name='random_int')(design_in)
        mixed = tkl.Concatenate()([mixed, reInt])

    x = tkl.Dense(1024 * input_shape[0] // 64 * input_shape[1] // 64, name='d_dense')(mixed)
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
    opt = tf.keras.optimizers.Adam()
    encoder = tf.keras.models.Model(encoder_in, encoder_out)
    decoder = tf.keras.models.Model([decoder_in, design_in], decoder_out)

    adversary = tf.keras.models.Model(adversary_in, adversary_out)
    adversary.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Make the adversary's layers untrainable when it's part of the autoencoder
    # It is still trainable in the original adversary model
    adversary.trainable = False

    autoencoder_in = tkl.Input(input_shape, name='ae_input')
    autoencoder_design_in = tkl.Input((n_groups,), name='ae_design_input')
    autoencoder_latent = encoder(autoencoder_in)
    autoencoder_out = decoder([autoencoder_latent, autoencoder_design_in])
    validity = adversary(autoencoder_latent)
    autoencoder = tf.keras.models.Model([autoencoder_in, autoencoder_design_in], 
                                        [autoencoder_out, validity])
    autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[recon_loss_weight, adv_loss_weight], optimizer=opt,
                        metrics=[['mean_squared_error'], ['accuracy']])
    
    return autoencoder, adversary, encoder