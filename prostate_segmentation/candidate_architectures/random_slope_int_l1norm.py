import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow_probability.python.layers as tfplayers
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.util as tfu
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import numpy as np
sys.path.append(os.path.abspath('/archive/bioinformatics/DLLab/KevinNguyen/src/MixedEffectsDL'))
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error

def make_posterior_fn(loc_init_range=1, scale_init_range=(0.01, 1)):

    def _posterior_fn(dtype, shape, name, trainable, add_variable_fn):
        loc = add_variable_fn(name=name + '_loc',
                            shape=shape,
                            initializer=tf.initializers.random_uniform(minval=-loc_init_range, maxval=loc_init_range),
                            dtype=dtype,
                            trainable=trainable)
        untransformed_scale = add_variable_fn(name=name + '_untransformed_scale',
                                            shape=shape[1:],
                                            initializer=tf.initializers.random_uniform(minval=scale_init_range[0], maxval=scale_init_range[1]),
                                            dtype=dtype,
                                            trainable=trainable)
        scale = tfu.DeferredTensor(untransformed_scale,
                                        lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)))
        dist = tfd.Normal(loc=loc, scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return _posterior_fn

def make_prior_fn(prior_scale=1):
    def _prior_mvn_fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates multivariate standard `Normal` distribution.

        Args:
            dtype: Type of parameter's event.
            shape: Python `list`-like representing the parameter's event shape.
            name: Python `str` name prepended to any created (or existing)
            `tf.Variable`s.
            trainable: Python `bool` indicating all created `tf.Variable`s should be
            added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
            add_variable_fn: `tf.get_variable`-like `callable` used to create (or
            access existing) `tf.Variable`s.

        Returns:
            Multivariate standard `Normal` distribution.
        """
        dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=prior_scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims)
    return _prior_mvn_fn    

class DenseRandomEffects(tfplayers.DenseFlipout):
    def __init__(self, 
                 units,
                 loc_init_range=1,
                 scale_init_range=(0.01, 1),
                 prior_scale=1,
                 activation=None,
                 activity_regularizer=None,
                 trainable=True,
                #  kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
                 kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
                 seed=None,
                 **kwargs):
        '''
        Probabilistic dense layer for learning single-level random effects. Based on 
        Tensorflow Probability's DenseFlipout layer. This layer receive as input 
        a one-hot encoded design matrix containing the group membership of each sample. 

        Args:
            units (int): number of neurons
            
            All other keyword arguments of DenseFlipout besides kernel_posterior_fn 
            and kernel_posterior_tensor_fn are available.
        '''
        super(DenseRandomEffects, self).__init__(units=units,
                                                activation=activation,
                                                activity_regularizer=activity_regularizer,
                                                trainable=trainable,
                                                kernel_posterior_fn=make_posterior_fn(loc_init_range, scale_init_range),
                                                kernel_posterior_tensor_fn=(lambda d: tf.math.reduce_mean(d.sample(100), axis=0)),
                                                kernel_prior_fn=make_prior_fn(prior_scale),
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_posterior_fn=None,
                                                bias_posterior_tensor_fn=None,
                                                bias_prior_fn=None,
                                                bias_divergence_fn=None,
                                                **kwargs)

def me_unet3d(tupInputShape, nGroups,
                loc_init_range=1,
                scale_init_min=0.1,
                scale_init_max=0.2,
                prior_scale=0.1,
                l1norm_weight=0.01):
    # Create model
    inputs = tkl.Input(tupInputShape)
    inputsRE = tkl.Input((nGroups,))
    conv1 = tkl.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tkl.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = tkl.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = tkl.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tkl.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tkl.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = tkl.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tkl.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = tkl.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = tkl.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tkl.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tkl.Dropout(0.5)(conv4)
    pool4 = tkl.MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = tkl.Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tkl.Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    # Insert random effects layers
    l1norm = tf.keras.regularizers.L1(l1=l1norm_weight)
    reSlope = DenseRandomEffects(1024, 
                                 loc_init_range=loc_init_range, 
                                 scale_init_range=(scale_init_min, scale_init_max),
                                 prior_scale=prior_scale,
                                 activity_regularizer=l1norm)(inputsRE)  
    reSlopeMult = tkl.Multiply()([conv5, reSlope])  
    mixed = tkl.Add()([reSlopeMult, conv5])
    # Random intercept? maybe a single bias term that gets reshaped into the feature map size?
    reInt = DenseRandomEffects(1,
                               loc_init_range=loc_init_range, 
                               scale_init_range=(scale_init_min, scale_init_max),
                               prior_scale=prior_scale,
                               activity_regularizer=l1norm)(inputsRE)
    arrFeatMapShape = np.array(tupInputShape[:-1]) * (0.5 ** 4)
    arrFeatMapShape = arrFeatMapShape.astype(int)
    def reshape_intercept(x):
        x1 = tf.reshape(x, (tf.shape(x)[0],) + (1, 1, 1, 1))
        return tf.tile(x1, (1, arrFeatMapShape[0], arrFeatMapShape[1], arrFeatMapShape[2], 1))
    
    reIntReshape = tkl.Lambda(reshape_intercept)(reInt)
    mixed = tkl.concatenate([mixed, reIntReshape], axis=-1)

    drop5 = tkl.Dropout(0.5)(mixed)

    up6 = tkl.Conv3D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling3D(size = (2,2,2))(drop5))
    merge6 = tkl.concatenate([drop4,up6], axis = 4)
    conv6 = tkl.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tkl.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tkl.Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling3D(size = (2,2,2))(conv6))
    merge7 = tkl.concatenate([conv3,up7], axis = 4)
    conv7 = tkl.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tkl.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tkl.Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling3D(size = (2,2,2))(conv7))
    merge8 = tkl.concatenate([conv2,up8], axis = 4)
    conv8 = tkl.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tkl.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = tkl.Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tkl.UpSampling3D(size = (2,2,2))(conv8))
    merge9 = tkl.concatenate([conv1,up9], axis = 4)
    conv9 = tkl.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tkl.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = tkl.Conv3D(1, 1, activation = 'sigmoid')(conv9)

    model = tf.keras.models.Model([inputs, inputsRE], conv10)

    return model