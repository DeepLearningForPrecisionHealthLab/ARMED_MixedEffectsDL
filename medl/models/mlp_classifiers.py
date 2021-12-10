"""Simple neural networks for classification
"""
import tensorflow as tf
import tensorflow.keras.layers as tkl
from .random_effects import RandomEffects

class BaseMLP(tf.keras.Model):
    def __init__(self, name: str='mlp', **kwargs):
        """Basic MLP with 3 hidden layers of 4 neurons each.

        Args:
            name (str, optional): Model name. Defaults to 'mlp'.
        """        
        super(BaseMLP, self).__init__(name=name, **kwargs)

        self.dense0 = tkl.Dense(4, activation='relu', name=name + '_dense0')
        self.dense1 = tkl.Dense(4, activation='relu', name=name + '_dense1')
        self.dense2 = tkl.Dense(4, activation='relu', name=name + '_dense2')
        self.dense_out = tkl.Dense(1, activation='sigmoid', name=name + '_dense_out')
        
    def call(self, inputs):
        
        x = self.dense0(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense_out(x)
        
        return x
    
class ClusterCovariateMLP(BaseMLP):
    """
    Basic MLP that concatenates the site membership design matrix to the data.
    """
    def call(self, inputs):
        x, z = inputs
        
        x = tf.concat((x, z), axis=1)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense_out(x)
    
        return x
class MLPActivations(tkl.Layer):
    def __init__(self, last_activation: str='sigmoid', name: str='mlp_activations', **kwargs):
        """Basic MLP with 3 hidden layers of 4 neurons each. In addition to the
        prediction, also returns the activation of each layer. Intended to be
        used within a domain adversarial model.

        Args: 
        last_activation (str, optional): Activation of output layer. Defaults to 
            'sigmoid'. 
        name (str, optional): Model name. Defaults to 'mlp_activations'.
        """        
        super(MLPActivations, self).__init__(name=name, **kwargs)

        self.dense0 = tkl.Dense(4, activation='relu', name=name + '_dense0')
        self.dense1 = tkl.Dense(4, activation='relu', name=name + '_dense1')
        self.dense2 = tkl.Dense(4, activation='relu', name=name + '_dense2')
        self.dense_out = tkl.Dense(1, activation=last_activation, name=name + '_dense_out')
        
    def call(self, inputs):
        
        x0 = self.dense0(inputs)
        x1 = self.dense1(x0)
        x2 = self.dense2(x1)
        out = self.dense_out(x2)
        
        return x0, x1, x2, out
    
    def get_config(self):
        return {}
    
class Adversary(tkl.Layer):
    def __init__(self,
                 n_clusters: int, 
                 layer_units: list=[8, 8, 4],
                 name: str='adversary',
                 **kwargs):
        """Adversarial classifier. 

        Args:
            n_clusters (int): number of clusters (classes)
            layer_units (list, optional): Neurons in each layer. Can be a list of any
                length. Defaults to [8, 8, 8].
            name (str, optional): Model name. Defaults to 'adversary'.
        """        
        
        super(Adversary, self).__init__(name=name, **kwargs)
        
        self.n_clusters = n_clusters
        self.layer_units = layer_units
        
        self.layers = []
        for iLayer, neurons in enumerate(layer_units):
            self.layers += [tkl.Dense(neurons, 
                                      activation='relu', 
                                      name=name + '_dense' + str(iLayer))]
            
        self.layers += [tkl.Dense(n_clusters, activation='softmax', name=name + '_dense_out')]
        
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def get_config(self):
        return {'n_clusters': self.n_clusters,
                'layer_units': self.layer_units}
        
class DomainAdversarialMLP(tf.keras.Model):
    def __init__(self, 
                 n_clusters: int, 
                 adversary_layer_units: list=[8, 8, 4], 
                 name: str='da_mlp', 
                 **kwargs):
        """Domain adversarial MLP classifier. The main model learns the classification
        task while the adversary prevents it from learning cluster-related features. 

        Args:
            n_clusters (int): Number of clusters.
            adversary_layer_units (list, optional): Neurons in each layer of the 
                adversary. Defaults to [8, 8, 4].
            name (str, optional): Model name. Defaults to 'da_mlp'.
        """        
        
        super(DomainAdversarialMLP, self).__init__(name=name, **kwargs)

        self.classifier = MLPActivations(name='mlp')
        self.adversary = Adversary(n_clusters=n_clusters, 
                                   layer_units=adversary_layer_units,
                                   name='adversary')
        
    def call(self, inputs):
        x, z = inputs
        classifier_outs = self.classifier(x)
        pred_class = classifier_outs[-1]
        activations = tf.concat(classifier_outs[:3], axis=1)
        pred_cluster = self.adversary(activations)
        
        return pred_class, pred_cluster
    
    def compile(self,
                loss_class=tf.keras.losses.BinaryCrossentropy(),
                loss_adv=tf.keras.losses.CategoricalCrossentropy(),
                metric_class=tf.keras.metrics.AUC(curve='PR', name='auprc'),
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_main=tf.keras.optimizers.Adam(lr=0.001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.001),
                loss_class_weight=1.0,
                loss_gen_weight=1.0,
                ):
        """Compile model with selected losses and metrics. Must be called before training.
        
        Loss weights apply to the main model: 
        total_loss = loss_class_weight * loss_class - loss_gen_weight * loss_adv

        Args:
            loss_class (loss, optional): Main classification loss. Defaults to 
                tf.keras.losses.BinaryCrossentropy().
            loss_adv (loss, optional): Adversary classification loss. Defaults to 
                tf.keras.losses.CategoricalCrossentropy().
            metric_class (metric, optional): Main classification metric. Defaults to 
                tf.keras.metrics.AUC(curve='PR', name='auprc').
            metric_adv (metric, optional): Adversary classification metric. Defaults to 
                tf.keras.metrics.CategoricalAccuracy(name='acc').
            opt_main (optimizer, optional): Main optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            opt_adversary (optimizer, optional): Adversary optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            loss_class_weight (float, optional): Classification loss weight. Defaults to 1.0.
            loss_gen_weight (float, optional): Generalization loss weight. Defaults to 1.0.
        """        
        
        super().compile()
        
        self.loss_class = loss_class
        self.loss_adv = loss_adv

        self.opt_main = opt_main
        self.opt_adversary = opt_adversary
        
        # Trackers for running mean of each loss
        self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')
        self.loss_adv_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        self.metric_class = metric_class
        self.metric_adv = metric_adv

        self.loss_class_weight = loss_class_weight
        self.loss_gen_weight = loss_gen_weight    
        
    @property
    def metrics(self):
        return [self.loss_class_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_class,
                self.metric_adv]
        
    def train_step(self, data):
        # Unpack data, including sample weights if provided
        if len(data) == 3:
            (data, clusters), labels, sample_weights = data
        else:
            (data, clusters), labels = data
            sample_weights = None
        
        # Get hidden layer activations from classifier and train the adversary    
        activations = tf.concat(self.classifier(data)[:-1], axis=1)
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(activations)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        # Train the main classifier
        with tf.GradientTape() as gt2:
            pred_class, pred_cluster = self((data, clusters))
            loss_class = self.loss_class(labels, pred_class, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
            total_loss = (self.loss_class_weight * loss_class) \
                - (self.loss_gen_weight * loss_adv)

        grads_class = gt2.gradient(total_loss, self.classifier.trainable_variables)
        self.opt_main.apply_gradients(zip(grads_class, self.classifier.trainable_variables))
        
        self.metric_class.update_state(labels, pred_class)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (data, clusters), labels = data
                        
        pred_class, pred_cluster = self((data, clusters))
        loss_class = self.loss_class(labels, pred_class)
        loss_adv = self.loss_adv(clusters, pred_cluster)
            
        total_loss = (self.loss_class_weight * loss_class) \
            - (self.loss_gen_weight * loss_adv)
                    
        self.metric_class.update_state(labels, pred_class)
        self.metric_adv.update_state(clusters, pred_cluster)
        
        self.loss_class_tracker.update_state(loss_class)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    

class RandomEffectsLinearSlopeIntercept(tkl.Layer):
    def __init__(self, 
                 slopes: int,
                 slope_posterior_init_scale: float=0.1, 
                 intercept_posterior_init_scale: float=0.1, 
                 slope_prior_scale: float=0.1,
                 intercept_prior_scale: float=0.1,
                 kl_weight: float=0.001, 
                 name: str='randomeffects', **kwargs):
        """Layer that learns a random linear slope and intercept. When called on an input
        (x, z), it returns a tuple of (f(random_slope(z) * x), random_intercept(z)).

        Args:
            slopes ([type]): dimensionality of the slopes (i.e. the number of features)
            slope_posterior_init_scale (float, optional): Scale for initializing slope 
                posterior means with a random normal distribution. Defaults to 0.1.
            intercept_posterior_init_scale (float, optional): Scale for initializing intercept 
                posterior means with a random normal distribution. Defaults to 0.1.
            slope_prior_scale (float, optional): Scale of slope prior distribution. Defaults to 0.1.
            intercept_prior_scale (float, optional): Intercept of intercept prior distribution. 
                Defaults to 0.1.
            kl_weight (float, optional): KL divergence loss weight. Defaults to 0.001.
            name (str, optional): Mode name. Defaults to 'randomeffects'.
        """        
        super(RandomEffectsLinearSlopeIntercept, self).__init__(name=name, **kwargs)
    
        self.slopes = slopes
        self.slope_posterior_init_scale = slope_posterior_init_scale
        self.intercept_posterior_init_scale = intercept_posterior_init_scale
        self.slope_prior_scale = slope_prior_scale
        self.intercept_prior_scale = intercept_prior_scale
        self.kl_weight= kl_weight
        
        self.re_slope = RandomEffects(slopes, 
                                      post_loc_init_scale=slope_posterior_init_scale,
                                      prior_scale=slope_prior_scale,
                                      kl_weight=kl_weight, name=name + '_re_slope')
        self.dense_out = tkl.Dense(1, name=name + '_re_out')
        
        self.re_int = RandomEffects(1, 
                                    post_loc_init_scale=intercept_posterior_init_scale,
                                    prior_scale=intercept_prior_scale,
                                    kl_weight=kl_weight, 
                                    name=name + '_re_int')
  
    def call(self, inputs, training=None):
        x, z = inputs        
        slope = self.re_slope(z, training=training)
        prod = self.dense_out(x * slope)
        intercept = self.re_int(z, training=training)
        
        return  prod, intercept
    
    def get_config(self):
        return {'slopes': self.slopes,
                'slope_posterior_init_scale': self.slope_posterior_init_scale,
                'intercept_posterior_init_scale': self.intercept_posterior_init_scale,
                'slope_prior_scale': self.slope_prior_scale,
                'intercept_prior_scale': self.intercept_prior_scale,
                'kl_weight': self.kl_weight}
        
class MixedEffectsMLP(DomainAdversarialMLP):
    def __init__(self, n_features: int, n_clusters: int, 
                 adversary_layer_units: list=[8, 8, 4], 
                 slope_posterior_init_scale: float=0.1, 
                 intercept_posterior_init_scale: float=0.1, 
                 slope_prior_scale: float=0.1,
                 intercept_prior_scale: float=0.1,
                 kl_weight: float=0.001,
                 name: str='me_mlp', 
                 **kwargs):
        """Mixed effects MLP classifier. Includes an adversarial classifier to 
        disentangle the predictive features from the cluster-specific features. 
        The cluster-specific features are then learned by a random effects layer. 
        
        This architecture includes linear random slopes (to be multiplied by the 
        input) and random intercept. The model output is 
        (fixed effect output) + (random slopes) * X + (random intercept)

        Args:
            n_features (int): Number of features.
            n_clusters (int): Number of clusters.
            adversary_layer_units (list, optional): Neurons in each layer of the 
                adversary. Defaults to [8, 8, 4].
            slope_posterior_init_scale (float, optional): Scale for initializing slope 
                posterior means with a random normal distribution. Defaults to 0.1.
            intercept_posterior_init_scale (float, optional): Scale for initializing intercept 
                posterior means with a random normal distribution. Defaults to 0.1.
            slope_prior_scale (float, optional): Scale of slope prior distribution. Defaults to 0.1.
            intercept_prior_scale (float, optional): Intercept of intercept prior distribution. 
                Defaults to 0.1.
            kl_weight (float, optional): KL divergence loss weight. Defaults to 0.001.
            name (str, optional): Model name. Defaults to 'me_mlp'.
        """        
    
        super(MixedEffectsMLP, self).__init__(n_clusters=n_clusters,
                                              adversary_layer_units=adversary_layer_units,
                                              name=name, **kwargs)
        self.classifier = MLPActivations(last_activation='linear', name='mlp')

        self.randomeffects = RandomEffectsLinearSlopeIntercept(
                        n_features,
                        slope_posterior_init_scale=slope_posterior_init_scale,
                        intercept_posterior_init_scale=intercept_posterior_init_scale,
                        slope_prior_scale=slope_prior_scale,
                        intercept_prior_scale=intercept_prior_scale,
                        kl_weight=kl_weight)

        
    def call(self, inputs, training=None):
        x, z = inputs
        fe_outs = self.classifier(x)
        pred_class_fe = tf.nn.sigmoid(fe_outs[-1])
                
        re_prod, re_int = self.randomeffects((x, z), training=training)
        pred_class_me = tf.nn.sigmoid(re_prod + re_int + pred_class_fe)     
        
        fe_activations = tf.concat(fe_outs[:3], axis=1)
        pred_cluster = self.adversary(fe_activations)
                
        return pred_class_me, pred_class_fe, pred_cluster
    
    def compile(self,
                loss_class=tf.keras.losses.BinaryCrossentropy(),
                loss_adv=tf.keras.losses.CategoricalCrossentropy(),
                metric_class_me=tf.keras.metrics.AUC(curve='PR', name='auprc'),
                metric_class_fe=tf.keras.metrics.AUC(curve='PR', name='auprc_fe'),
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_main=tf.keras.optimizers.Adam(lr=0.001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.001),
                loss_class_me_weight=1.0,
                loss_class_fe_weight=1.0,
                loss_gen_weight=1.0,
                ):
        """Compile model with selected losses and metrics. Must be called before training.
        
        Loss weights apply to the main model: 
        total_loss = loss_class_me_weight * loss_class_me + loss_class_fe_weight * loss_class_fe
            - loss_gen_weight * loss_adv

        Args:
            loss_class (loss, optional): Main classification loss. This applies to both the 
                mixed and fixed effects-based classifications. Defaults to 
                tf.keras.losses.BinaryCrossentropy().
            loss_adv (loss, optional): Adversary classification loss. Defaults to 
                tf.keras.losses.CategoricalCrossentropy().
            metric_class_me ([type], optional): Metric for classification using mixed effects. 
                Defaults to tf.keras.metrics.AUC(curve='PR', name='auprc').
            metric_class_fe ([tye], optional): Metric for classification using fixed effects. 
                Defaults to tf.keras.metrics.AUC(curve='PR', name='auprc_fe').
            metric_adv (metric, optional): Adversary classification metric. Defaults to 
                tf.keras.metrics.CategoricalAccuracy(name='acc').
            opt_main (optimizer, optional): Main optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            opt_adversary (optimizer, optional): Adversary optimizer. Defaults to 
                tf.keras.optimizers.Adam(lr=0.001).
            loss_class_me_weight (float, optional): Weight for classification using mixed 
                effects. Defaults to 1.0.
            loss_class_fe_weight (float, optional): Weight for classification using fixed 
                effects. Defaults to 1.0.
            loss_gen_weight (float, optional): Generalization loss weight. Defaults to 1.0.
        """  
        
        super().compile()
        
        self.loss_class = loss_class
        self.loss_adv = loss_adv

        self.opt_main = opt_main
        self.opt_adversary = opt_adversary
        
        # Loss trackers
        self.loss_class_me_tracker = tf.keras.metrics.Mean(name='class_me_loss')
        self.loss_class_fe_tracker = tf.keras.metrics.Mean(name='class_fe_loss')
        self.loss_adv_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        self.metric_class_me = metric_class_me
        self.metric_class_fe = metric_class_fe
        self.metric_adv = metric_adv

        self.loss_class_me_weight = loss_class_me_weight
        self.loss_class_fe_weight = loss_class_fe_weight
        self.loss_gen_weight = loss_gen_weight    
        
        # Unneeded
        del self.loss_class_tracker, self.loss_class_weight, self.metric_class
        
    @property
    def metrics(self):
        return [self.loss_class_me_tracker,
                self.loss_class_fe_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_class_me,
                self.metric_class_fe,
                self.metric_adv]
        
    def train_step(self, data):
        # Unpack data, including sample weights if provided
        if len(data) == 3:
            (data, clusters), labels, sample_weights = data
        else:
            (data, clusters), labels = data
            sample_weights = None
        
        # Get hidden layer activations from classifier and train the adversary       
        activations = tf.concat(self.classifier(data)[:-1], axis=1)
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(activations)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        # Train the main classifier 
        with tf.GradientTape() as gt2:
            pred_class_me, pred_class_fe, pred_cluster = self((data, clusters), training=True)
            loss_class_me = self.loss_class(labels, pred_class_me, sample_weight=sample_weights)
            loss_class_fe = self.loss_class(labels, pred_class_fe, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
            total_loss = (self.loss_class_me_weight * loss_class_me) \
                + (self.loss_class_fe_weight * loss_class_fe) \
                - (self.loss_gen_weight * loss_adv) \
                + self.randomeffects.losses

        lsVars = self.classifier.trainable_variables + self.randomeffects.trainable_variables
        grads_class = gt2.gradient(total_loss, lsVars)
        self.opt_main.apply_gradients(zip(grads_class, lsVars))
        
        self.metric_class_me.update_state(labels, pred_class_me)
        self.metric_class_fe.update_state(labels, pred_class_fe)
        self.loss_class_me_tracker.update_state(loss_class_me)
        self.loss_class_fe_tracker.update_state(loss_class_fe)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (data, clusters), labels = data
                        
        pred_class_me, pred_class_fe, pred_cluster = self((data, clusters), training=False)
        loss_class_me = self.loss_class(labels, pred_class_me)
        loss_class_fe = self.loss_class(labels, pred_class_fe)
        loss_adv = self.loss_adv(clusters, pred_cluster)
            
        total_loss = (self.loss_class_me_weight * loss_class_me) \
                + (self.loss_class_fe_weight * loss_class_fe) \
                - (self.loss_gen_weight * loss_adv) \
                + self.randomeffects.losses
                    
        self.metric_class_me.update_state(labels, pred_class_me)
        self.metric_class_fe.update_state(labels, pred_class_fe)
        self.metric_adv.update_state(clusters, pred_cluster)
        
        self.loss_class_me_tracker.update_state(loss_class_me)
        self.loss_class_fe_tracker.update_state(loss_class_fe)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
        
        
class MixedEffectsMLPNonlinearSlope(MixedEffectsMLP):
    def __init__(self, n_features: int, n_clusters: int, 
                 adversary_layer_units: list=[8, 8, 4], 
                 slope_posterior_init_scale: float=0.1, 
                 intercept_posterior_init_scale: float=0.1, 
                 slope_prior_scale: float=0.1,
                 intercept_prior_scale: float=0.1,
                 kl_weight: float=0.001,
                 name: str='me_mlp', 
                 **kwargs):
        """Mixed effects MLP classifier. Includes an adversarial classifier to 
        disentangle the predictive features from the cluster-specific features. 
        The cluster-specific features are then learned by a random effects layer. 

        This architecture includes nonlinear random slopes (to be multiplied by the 
        penultimate layer output of the fixed effects submodel) and random intercept. 
        The model output is 
        (fixed effect output) + (random slopes) * (penultimate FE layer output) + (random intercept)

        Args:
            n_features (int): Number of features.
            n_clusters (int): Number of clusters.
            adversary_layer_units (list, optional): Neurons in each layer of the 
                adversary. Defaults to [8, 8, 4].
            slope_posterior_init_scale (float, optional): Scale for initializing slope 
                posterior means with a random normal distribution. Defaults to 0.1.
            intercept_posterior_init_scale (float, optional): Scale for initializing intercept 
                posterior means with a random normal distribution. Defaults to 0.1.
            slope_prior_scale (float, optional): Scale of slope prior distribution. Defaults to 0.1.
            intercept_prior_scale (float, optional): Intercept of intercept prior distribution. 
                Defaults to 0.1.
            kl_weight (float, optional): KL divergence loss weight. Defaults to 0.001.
            name (str, optional): Model name. Defaults to 'me_mlp'.
        """       
        del n_features # unused
    
        super(MixedEffectsMLP, self).__init__(n_clusters=n_clusters,
                                              adversary_layer_units=adversary_layer_units,
                                              name=name, **kwargs)
        self.classifier = MLPActivations(last_activation='linear', name='mlp')

        self.randomeffects = RandomEffectsLinearSlopeIntercept(
                        slopes=4,
                        slope_posterior_init_scale=slope_posterior_init_scale,
                        intercept_posterior_init_scale=intercept_posterior_init_scale,
                        slope_prior_scale=slope_prior_scale,
                        intercept_prior_scale=intercept_prior_scale,
                        kl_weight=kl_weight)

    def call(self, inputs, training=None):
        x, z = inputs
        fe_outs = self.classifier(x)
        pred_class_fe = tf.nn.sigmoid(fe_outs[-1])
        
        # Penultimate FE layer output
        fe_latents = fe_outs[-2]        
        
        re_prod, re_int = self.randomeffects((fe_latents, z), training=training)
        pred_class_me = tf.nn.sigmoid(re_prod + re_int + pred_class_fe)     
        
        fe_activations = tf.concat(fe_outs[:3], axis=1)
        pred_cluster = self.adversary(fe_activations)
                
        return pred_class_me, pred_class_fe, pred_cluster