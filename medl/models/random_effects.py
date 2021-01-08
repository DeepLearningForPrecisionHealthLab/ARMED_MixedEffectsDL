'''
WIP: probabilistic layers for implementing random effects modeling
'''

import tensorflow as tf
import tensorflow_probability.python.layers as tfplayers
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.util as tfu
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import numpy as np

def posterior_fn(dtype, shape, name, trainable, add_variable_fn):
    loc = add_variable_fn(name=name + '_loc',
                        shape=shape,
                        initializer=tf.initializers.random_uniform(minval=-1, maxval=1),
                        dtype=dtype,
                        trainable=trainable)

    untransformed_scale = add_variable_fn(name=name + '_untransformed_scale',
                                        shape=[1],
                                        initializer=tf.initializers.random_uniform(minval=0.01, maxval=1),
                                        dtype=dtype,
                                        trainable=trainable)
    scale = tfu.DeferredTensor(untransformed_scale,
                                    lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)))
    dist = tfd.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

class DenseRandomEffects(tfplayers.DenseFlipout):
    def __init__(self, 
                 units,
                 activation=None,
                 activity_regularizer=None,
                 trainable=True,
                 kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
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
                                                kernel_posterior_fn=posterior_fn,
                                                kernel_posterior_tensor_fn=(lambda d: tf.math.reduce_mean(d.sample(100), axis=0)),
                                                kernel_prior_fn=kernel_prior_fn,
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_posterior_fn=None,
                                                bias_posterior_tensor_fn=None,
                                                bias_prior_fn=None,
                                                bias_divergence_fn=None,
                                                **kwargs)
