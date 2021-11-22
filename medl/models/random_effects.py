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
                                        tpd.Independent(
                                            tpd.Normal(loc=tf.zeros(n), scale=prior_scale),
                                            reinterpreted_batch_ndims=1))
                                    ])
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

class NamedVariableLayer(tpl.VariableLayer):
    def __init__(self,
                shape,
                dtype=None,
                activation=None,
                initializer='zeros',
                regularizer=None,
                constraint=None,
                name=None,
                **kwargs) -> None:
        '''
        Subclass of VariableLayer that simply adds the capability to name the
        variables. This is needed to prevent name collisions when saving model
        weights; the original VariableLayer hardcodes the variable name to
        'constant' for every instance.

        '''

        super(tpl.VariableLayer, self).__init__(**kwargs)

        self.activation = tf.keras.activations.get(activation)
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)
        self.shape = shape

        shape = tf.get_static_value(shape)
        if shape is None:
            raise ValueError('Shape must be known statically.')
        shape = np.array(shape, dtype=np.int32)
        ndims = len(shape.shape)
        if ndims > 1:
            raise ValueError('Shape must be scalar or vector.')
        shape = shape.reshape(-1)  # Ensures vector shape.

        self._var = self.add_weight(
            name,
            shape=shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            dtype=dtype,
            trainable=kwargs.get('trainable', True))
        
    def get_config(self):
        return {'shape': self.shape}    
    
'''
Gamma posterior and prior distributions. This distribution has parameters alpha (concentration) and beta (rate, or inverse scale). 
'''
def make_deterministic_posterior_fn():
    def _re_posterior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        # There are n variables containing the mean of each weight and n variables
        # containing the shared s.d. for all weights
        initializer = tf.keras.initializers.RandomUniform(minval=0.1, maxval=1.0)

        return tf.keras.Sequential([NamedVariableLayer(n, dtype=dtype, initializer=initializer, constraint='non_neg', name='posterior'),
                                    tpl.DistributionLambda(lambda t: tpd.Deterministic(loc=t, rtol=0.00001))])
    return _re_posterior_fn

def make_gamma_prior_fn():
    def _prior_fn(kernel_size, bias_size=0, dtype=None):
        initializer = tf.keras.initializers.RandomUniform(minval=5, maxval=10)
        return tf.keras.Sequential([NamedVariableLayer(1, dtype=dtype, initializer=initializer, constraint='non_neg', name='gamma'),
                                    tpl.DistributionLambda(lambda t: 
                                        tpd.Gamma(concentration=1.0, rate=t))])
    return _prior_fn

class GammaRandomEffects(RandomEffects):
    def __init__(self, units=1, kl_weight=0.001, l1_weight=None, name=None) -> None:
        self.units = units
        self.kl_weight = kl_weight
        self.l1_weight = l1_weight
        
        posterior = make_deterministic_posterior_fn()
        prior = make_gamma_prior_fn()
                
        super(RandomEffects, self).__init__(units, posterior, prior, use_bias=False,
                                            kl_weight=kl_weight,
                                            kl_use_exact=True,
                                            name=name)
        
    def get_config(self):
        return {'units': self.units, 
                'kl_weight': self.kl_weight, 
                'l1_weight': self.l1_weight}
        