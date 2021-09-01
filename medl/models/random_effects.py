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

