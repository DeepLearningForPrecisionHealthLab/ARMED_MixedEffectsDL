import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl
from tensorflow_probability import layers as tpl
from tensorflow_probability import distributions as tpd

from tensorflow_addons.layers import InstanceNormalization

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
        
    def call(self, inputs, training=None):
        
        if training == False:
            # In testing mode, use the posterior means 
            if self._posterior.built == False:
                self._posterior.build(inputs.shape)
            if self._prior.built == False:
                self._prior.build(inputs.shape)
            
            # First half of weights contains the posterior means
            nWeights = self.weights[0].shape[0]
            w = self.weights[0][:(nWeights // 2)]
                        
            prev_units = self.input_spec.axes[-1]

            kernel = tf.reshape(w, shape=tf.concat([
                tf.shape(w)[:-1],
                [prev_units, self.units],
            ], axis=0))
            outputs = tf.matmul(inputs, kernel)

            if self.activation is not None:
                outputs = self.activation(outputs)  # pylint: disable=not-callable
        else:
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
                                    tpl.DistributionLambda(lambda t: tpd.VectorDeterministic(loc=t, rtol=0.00001))])
    return _re_posterior_fn

def make_gamma_prior_fn():
    def _prior_fn(kernel_size, bias_size=0, dtype=None):
        initializer = tf.keras.initializers.RandomUniform(minval=5, maxval=10)
        n = kernel_size + bias_size
        return tf.keras.Sequential([NamedVariableLayer(n, dtype=dtype, initializer=initializer, 
                                                       constraint='non_neg', name='rate'),
                                    tpl.DistributionLambda(lambda t: 
                                        tpd.Independent(tpd.Gamma(concentration=1.0, rate=t),
                                                        reinterpreted_batch_ndims=1))])
    return _prior_fn

class GammaRandomEffects(RandomEffects):
    def __init__(self, units=1, kl_weight=0.001, l1_weight=None, name=None) -> None:
        self.units = units
        self.kl_weight = kl_weight
        self.l1_weight = l1_weight
        
        # posterior = make_deterministic_posterior_fn()
        posterior = make_posterior_fn(0.1, 0.05, 0.05)
        # Something about this prior prevents model from compiling. Works fine
        # when using the normal prior and deterministic posterior.
        prior = make_gamma_prior_fn()
                        
        super(RandomEffects, self).__init__(units, posterior, prior, use_bias=False,
                                            kl_weight=kl_weight,
                                            kl_use_exact=False,
                                            name=name)
        
    def call(self, inputs, training=None):
        
        outputs = super(RandomEffects, self).call(inputs)
        
        if self.l1_weight:
            postmeans = self.weights[0]
            self.add_loss(self.l1_weight * tf.reduce_sum(tf.abs(postmeans)))
        
        return outputs
    
    def get_config(self):
        return {'units': self.units, 
                'kl_weight': self.kl_weight, 
                'l1_weight': self.l1_weight}
        

# def make_poisson_prior_fn(init_rate=1):
#     def _prior_fn(kernel_size, bias_size=0, dtype=None):
#         return tf.keras.Sequential([tpl.VariableLayer(1, dtype=dtype, 
#                                                       initializer=tf.initializers.Constant(init_rate), 
#                                                       constraint='non_neg'),
#                                     tpl.DistributionLambda(lambda t: tpd.Poisson(rate=t))])
#     return _prior_fn

# class PoissonRandomEffects(RandomEffects):
#     def __init__(self, units=1, kl_weight=0.001, l1_weight=None, name=None) -> None:
#         self.units = units
#         self.kl_weight = kl_weight
#         self.l1_weight = l1_weight
        
#         posterior = make_deterministic_posterior_fn()
#         prior = make_poisson_prior_fn()
                        
#         super(RandomEffects, self).__init__(units, posterior, prior, use_bias=False,
#                                             kl_weight=kl_weight,
#                                             kl_use_exact=False,
#                                             name=name)
        
#     def call(self, inputs, training=None):
        
#         outputs = super(RandomEffects, self).call(inputs)
        
#         if self.l1_weight:
#             postmeans = self.weights[0]
#             self.add_loss(self.l1_weight * tf.reduce_sum(tf.abs(postmeans)))
        
#         return outputs
    
#     def get_config(self):
#         return {'units': self.units, 
#                 'kl_weight': self.kl_weight, 
#                 'l1_weight': self.l1_weight}

class ClusterScaleBiasBlock(tkl.Layer):
    
    def __init__(self,
                 n_features, 
                 post_loc_init_scale=0.25,
                 prior_scale=0.25,
                 gamma_dist=False,
                 kl_weight=0.001,
                 name='cluster', 
                 **kwargs):
        super(ClusterScaleBiasBlock, self).__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.post_loc_init_scale = post_loc_init_scale
        self.prior_scale = prior_scale
        self.gamma_dist = gamma_dist
        
        self.kl_weight = kl_weight
        
        self.instance_norm = InstanceNormalization(center=True, 
                                                   scale=True, 
                                                   name=name + '_instance_norm')

        if gamma_dist:
            # Separate layer needed per feature, so that each can learn its own 
            # distribution
            # self.gammas = []
            # self.betas = []
            # for i in range(n_features):
            #     self.gammas += [GammaRandomEffects(units=1, kl_weight=kl_weight,
            #                                        name=name + f'_gammas_{i}')]
            #     self.betas += [GammaRandomEffects(units=1, kl_weight=kl_weight,
            #                                       name=name + f'_betas_{i}')]
        
            self.gammas = GammaRandomEffects(n_features, 
                                        kl_weight=kl_weight,
                                        name=name + '_gammas')
            self.betas = GammaRandomEffects(n_features, 
                                    kl_weight=kl_weight,
                                    name=name + '_betas')
        else:
            self.gammas = RandomEffects(n_features, 
                                        post_loc_init_scale=post_loc_init_scale,
                                        post_scale_init_min=0.01, 
                                        post_scale_init_range=0.005, 
                                        prior_scale=prior_scale, 
                                        kl_weight=kl_weight,
                                        name=name + '_gammas')
            self.betas = RandomEffects(n_features, 
                                    post_loc_init_scale=post_loc_init_scale,
                                    post_scale_init_min=0.01, 
                                    post_scale_init_range=0.005, 
                                    prior_scale=prior_scale, 
                                    kl_weight=kl_weight,
                                    name=name + '_betas')
        # self.multiply = tkl.Multiply(name=name + '_mult')
        # self.add = tkl.Add(name=name + '_add')

    def call(self, inputs, training=None):
        x, z = inputs
        x = self.instance_norm(x)
        # if self.gamma_dist:
        #     g = tf.concat([gamma(z) for gamma in self.gammas], axis=-1)
        #     b = tf.concat([beta(z) for beta in self.betas], axis=-1)
        # else:
        g = self.gammas(z, training=training)
        b = self.betas(z, training=training)    
        # Ensure shape is batch_size x 1 x 1 x n_features
        if len(tf.shape(x)) > 2:
            new_dims = len(tf.shape(x)) - 2
            g = tf.reshape(g, [-1] + [1] * new_dims + [self.n_features])
            b = tf.reshape(b, [-1] + [1] * new_dims + [self.n_features])
        
        # m = self.multiply((x, g))
        # s = self.add((m, b))
        m = x * (1 + g)
        s = m + b
        return s
    
    def get_config(self):
        return {'post_loc_init_scale': self.post_loc_init_scale,
                'prior_scale': self.prior_scale,
                'gamma_dist': self.gamma_dist,
                'kl_weight': self.kl_weight}       