import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl
from tensorflow_probability import layers as tpl
from tensorflow_probability import distributions as tpd

def make_posterior_fn(post_loc_init_scale, post_scale_init_min, post_scale_init_range):
    def _re_posterior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        # There are n variables containing the mean of each weight and n variables
        # containing the shared s.d. for all weights
        initializer = tpl.BlockwiseInitializer([tf.keras.initializers.RandomNormal(mean=0, 
                                                                                   stddev=post_loc_init_scale), 
                                                tf.keras.initializers.RandomUniform(minval=post_scale_init_min, 
                                                                                    maxval=post_scale_init_min \
                                                                                        + post_scale_init_range),
                                                ],
                                            sizes=[n, n])

        return tf.keras.Sequential([tpl.VariableLayer(n + n, dtype=dtype, initializer=initializer),
                                    tpl.DistributionLambda(lambda t: tpd.Independent(
                                    tpd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(t[..., n:])),
                                    reinterpreted_batch_ndims=1))
                                ])
    return _re_posterior_fn


def make_fixed_prior_fn(prior_scale):
    def _prior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([tpl.DistributionLambda(lambda t: 
                                        tpd.Normal(loc=tf.zeros(n), scale=prior_scale))])
    return _prior_fn

def make_trainable_prior_fn(prior_scale):
    def _prior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        initializer = tf.initializers.Constant(prior_scale)
        return tf.keras.Sequential([tpl.VariableLayer(n, dtype=dtype, initializer=initializer),
                                    tpl.DistributionLambda(lambda t: 
                                        tpd.Normal(loc=tf.zeros(n), scale=1e-5 + tf.nn.softplus(t)))])
    return _prior_fn

class RandomEffects(tpl.DenseVariational):
    def __init__(self, 
                 units=1, 
                 post_loc_init_scale=0.05, 
                 post_scale_init_min=0.05,
                 post_scale_init_range=0.05,
                 prior_scale=0.05,
                 kl_weight=0.001,
                 l1_weight=None,
                 name=None) -> None:
        
        self.kl_weight = kl_weight
        self.l1_weight = l1_weight
        
        # Compute inverse softplus
        fPostScaleMin = np.log(np.exp(post_scale_init_min) - 1)
        fPostScaleRange = np.log(np.exp(post_scale_init_range) - 1)
        
        posterior = make_posterior_fn(post_loc_init_scale, fPostScaleMin, fPostScaleRange)
        prior = make_fixed_prior_fn(prior_scale)
        # prior = make_trainable_prior_fn(prior_scale)
        
        super().__init__(units, posterior, prior, use_bias=False,
                         kl_weight=kl_weight,
                         name=name)
        
    def call(self, inputs):
        
        outputs = super().call(inputs)
        
        if self.l1_weight:
            # First half of weights contains the posterior means
            nWeights = self.weights[0].shape[0]
            postmeans = self.weights[0][:(nWeights // 2)]
            
            self.add_loss(self.l1_weight * tf.reduce_sum(tf.abs(postmeans)))
        
        return outputs

def unet_mixedeffects(random_effects_size, 
                      pretrained_weights=None, 
                      input_size=(256,256,1),
                      post_loc_init_scale=0.05,
                      post_scale_init_min=0.05,
                      post_scale_init_range=0.05,
                      prior_scale=0.05,
                      kl_weight=0.001,
                      l1_weight=None):
    
    inputs = tkl.Input(input_size)
    inputsRE = tkl.Input(random_effects_size)
    conv1 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = tkl.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tkl.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = tkl.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tkl.Dropout(0.5)(conv4)
    pool4 = tkl.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tkl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tkl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = tkl.Dropout(0.5)(conv5)

    # Insert random effects layers
    # latent_shape = np.ones((3,))
    # latent_shape[:2] = np.array(input_size[:-1]) * (0.5 ** 4)
    
    reSlope = RandomEffects(256, 
                            post_loc_init_scale=post_loc_init_scale, 
                            post_scale_init_min=post_scale_init_min,
                            post_scale_init_range=post_scale_init_range,
                            prior_scale=prior_scale,
                            kl_weight=kl_weight,
                            l1_weight=l1_weight,
                            name='RE_slope'
                            )(inputsRE)
    reSlopeMult = tkl.Multiply()([conv5, reSlope]) # -> n x m x 256
    mixed = tkl.Concatenate(axis=-1)([drop5, reSlopeMult]) # -> n x m x 512
    # mixed = tkl.Add()([drop5, reSlopeMult])

    up6 = tkl.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(mixed))
    merge6 = tkl.concatenate([drop4,up6], axis = 3)
    conv6 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tkl.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(conv6))
    merge7 = tkl.concatenate([conv3,up7], axis = 3)
    conv7 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tkl.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(conv7))
    merge8 = tkl.concatenate([conv2,up8], axis = 3)
    conv8 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = tkl.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(conv8))
    merge9 = tkl.concatenate([conv1,up9], axis = 3)
    conv9 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tkl.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = tkl.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = tf.keras.Model([inputs, inputsRE], conv10)

    if pretrained_weights:
    	model.load_weights(pretrained_weights)

    return model

def unet_mixedeffects_forked(random_effects_size, 
                                pretrained_weights=None, 
                                input_size=(256,256,1),
                                post_loc_init_scale=0.05,
                                post_scale_init_min=0.05,
                                post_scale_init_range=0.05,
                                prior_scale=0.05,
                                kl_weight=0.001,
                                l1_weight=None):
    
    # Downsampling branch
    inputsDown = tkl.Input(input_size, name='input_down')
    
    conv1 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv1a')(inputsDown)
    conv1 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv1b')(conv1)
    shape1 = np.array(input_size, dtype=int)
    shape1[-1] = 16
    pool1 = tkl.MaxPooling2D(pool_size=(2, 2), name='down_pool1')(conv1)
    
    conv2 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv2a')(pool1)
    conv2 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv2b')(conv2)
    shape2 = shape1 // 2
    shape2[-1] = 32
    pool2 = tkl.MaxPooling2D(pool_size=(2, 2), name='down_pool2')(conv2)
        
    conv3 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv3a')(pool2)
    conv3 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv3b')(conv3)
    shape3 = shape2 // 2
    shape3[-1] = 64
    pool3 = tkl.MaxPooling2D(pool_size=(2, 2), name='down_pool3')(conv3)
        
    conv4 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv4a')(pool3)
    conv4 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv4b')(conv4)
    drop4 = tkl.Dropout(0.5, name='down_dropout4')(conv4)
    shape4 = shape3 // 2
    shape4[-1] = 128
    pool4 = tkl.MaxPooling2D(pool_size=(2, 2), name='down_pool4')(drop4)
    
    conv5 = tkl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv5a')(pool4)
    conv5 = tkl.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='down_conv5b')(conv5)
    shape5 = shape4 // 2
    shape5[-1] = 256
    
    modelDown = tf.keras.Model(inputsDown, [conv1, conv2, conv3, drop4, conv5], name='down_branch')
     
    # Upsampling branch
    inputsUp1 = tkl.Input(shape1, name='input_up1')
    inputsUp2 = tkl.Input(shape2, name='input_up2')
    inputsUp3 = tkl.Input(shape3, name='input_up3')
    inputsUp4 = tkl.Input(shape4, name='input_up4')
    inputsLatent = tkl.Input(shape5, name='input_latent')
    
    drop5 = tkl.Dropout(0.5)(inputsLatent)    
    up6 = tkl.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(drop5))
    merge6 = tkl.concatenate([inputsUp4,up6], axis = 3)
    conv6 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tkl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tkl.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(conv6))
    merge7 = tkl.concatenate([inputsUp3,up7], axis = 3)
    conv7 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tkl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tkl.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(conv7))
    merge8 = tkl.concatenate([inputsUp2,up8], axis = 3)
    conv8 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tkl.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = tkl.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling2D(size = (2,2))(conv8))
    merge9 = tkl.concatenate([inputsUp1,up9], axis = 3)
    conv9 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tkl.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = tkl.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv10 = tkl.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    modelUp = tf.keras.Model([inputsUp1, inputsUp2, inputsUp3, inputsUp4, inputsLatent], conv9)
        
    # Put everything together
    inputsX = tkl.Input(input_size)
    inputsZ = tkl.Input(random_effects_size)
    outDown1, outDown2, outDown3, outDown4, outDownLatents = modelDown(inputsX)
    featuresFE = modelUp((outDown1, outDown2, outDown3, outDown4, outDownLatents))
    # featuresRE = modelUpRE((latents, inputsZ))
    
    reSlope = RandomEffects(256, 
                            post_loc_init_scale=post_loc_init_scale, 
                            post_scale_init_min=post_scale_init_min,
                            post_scale_init_range=post_scale_init_range,
                            prior_scale=prior_scale,
                            kl_weight=kl_weight,
                            l1_weight=l1_weight,
                            name='RE_slope'
                            )(inputsZ)
    reSlopeMult = tkl.Multiply()([outDownLatents, reSlope]) # -> n x m x 256
    
    featuresRE = modelUp((outDown1, outDown2, outDown3, outDown4, reSlopeMult))
     
    featuresMixed = tkl.Concatenate(axis=-1)([featuresFE, featuresRE])
    
    outputMain = tkl.Conv2D(1, 1, activation='sigmoid')(featuresMixed)
    
    modelMain = tf.keras.Model((inputsX, inputsZ), outputMain)
    
    if pretrained_weights:
    	modelMain.load_weights(pretrained_weights)

    return modelMain