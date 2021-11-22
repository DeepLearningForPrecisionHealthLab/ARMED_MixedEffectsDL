import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow_addons.layers import InstanceNormalization
from medl.models.random_effects import RandomEffects


class TiedConv2DTranspose(tkl.Conv2DTranspose):
    def __init__(self, 
                 source_layer,
                 filters, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding='valid', 
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Conv2DTranspose layer that shares kernels with a given Conv2D layer.
        
        (The bias tensor is not shared as the dimensionality of the output is
        inherently different.)

        Args: 
            source_layer (Conv2D): Conv2D layer
        """        
        self.source_layer = source_layer
                
        super().__init__(filters,
                         kernel_size,
                         strides=strides, 
                         padding=padding, 
                         output_padding=output_padding, 
                         data_format=data_format, 
                         dilation_rate=dilation_rate, 
                         activation=activation, 
                         use_bias=use_bias, 
                         kernel_initializer=kernel_initializer, 
                         bias_initializer=bias_initializer, 
                         kernel_regularizer=kernel_regularizer, 
                         bias_regularizer=bias_regularizer, 
                         activity_regularizer=activity_regularizer, 
                         kernel_constraint=kernel_constraint, 
                         bias_constraint=bias_constraint, 
                         **kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input '
                            'shape: ' + str(input_shape))
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                            'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        # kernel_shape = self.kernel_size + (self.filters, input_dim)

        # Link to filter kernels from the source conv layer
        self.kernel = self.source_layer.weights[0]
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

class Encoder(tkl.Layer):
        
    def __init__(self, 
                 n_latent_dims=56, 
                 layer_filters=[64, 128, 256, 512, 1024, 1024],
                 name='encoder', **kwargs):
        """Transforms 2D image into a compressed vector representation. Contains
        6x 2D strided convolutional layers.

        Args: 
            n_latent_dims (int, optional): Size of compressed representation
                output. Defaults to 56. 
            layer_filters (list, optional): Filters per convolutional layer. 
                Defaults to [64, 128, 256, 512, 1024, 1024].
            name (str, optional): Name. Defaults to 'encoder'.
        """        
        super(Encoder, self).__init__(name=name, **kwargs)
        
        self.n_latent_dims = n_latent_dims
        self.layer_filters = layer_filters
        
        self.conv0 = tkl.Conv2D(layer_filters[0], 4, strides=(2, 2), padding='same', name=name + '_conv0')
        self.bn0 = tkl.BatchNormalization(name=name+ '_bn0')
        self.prelu0 = tkl.PReLU(name=name + '_prelu0')
        
        self.conv1 = tkl.Conv2D(layer_filters[1], 4, strides=(2, 2), padding='same', name=name + '_conv1')
        self.bn1 = tkl.BatchNormalization(name=name+ '_bn1')
        self.prelu1 = tkl.PReLU(name=name + '_prelu1')
        
        self.conv2 = tkl.Conv2D(layer_filters[2], 4, strides=(2, 2), padding='same', name=name + '_conv2')
        self.bn2 = tkl.BatchNormalization(name=name+ '_bn2')
        self.prelu2 = tkl.PReLU(name=name + '_prelu2')
        
        self.conv3 = tkl.Conv2D(layer_filters[3], 4, strides=(2, 2), padding='same', name=name + '_conv3')
        self.bn3 = tkl.BatchNormalization(name=name+ '_bn3')
        self.prelu3 = tkl.PReLU(name=name + '_prelu3')
        
        self.conv4 = tkl.Conv2D(layer_filters[4], 4, strides=(2, 2), padding='same', name=name + '_conv4')
        self.bn4 = tkl.BatchNormalization(name=name+ '_bn4')
        self.prelu4 = tkl.PReLU(name=name + '_prelu4')
        
        self.conv5 = tkl.Conv2D(layer_filters[5], 4, strides=(2, 2), padding='same', name=name + '_conv5')
        self.bn5 = tkl.BatchNormalization(name=name+ '_bn5')
        self.prelu5 = tkl.PReLU(name=name + '_prelu5')
        
        self.flatten = tkl.Flatten(name=name + '_flatten')
        self.dense = tkl.Dense(n_latent_dims, name=name + '_latent')
        self.bn_out = tkl.BatchNormalization(name=name + '_output')
        
    def call(self, inputs, training=None):
        x0 = self.conv0(inputs)
        x0 = self.bn0(x0, training=training)
        x0 = self.prelu0(x0)
        
        x1 = self.conv1(x0)
        x1 = self.bn1(x1, training=training)
        x1 = self.prelu1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.bn2(x2, training=training)
        x2 = self.prelu2(x2)
        
        x3 = self.conv3(x2)
        x3 = self.bn3(x3, training=training)
        x3 = self.prelu3(x3)
        
        x4 = self.conv4(x3)
        x4 = self.bn4(x4, training=training)
        x4 = self.prelu4(x4)
        
        x5 = self.conv5(x4)
        x5 = self.bn5(x5, training=training)
        x5 = self.prelu5(x5)
        
        latent = self.flatten(x5)
        latent = self.dense(latent)
        latent = self.bn_out(latent)
        
        return x0, x1, x2, x3, x4, x5, latent
        
    def get_config(self):
        return {'n_latent_dims': self.n_latent_dims,
                'layer_filters': self.layer_filters}
        
class TiedDecoder(tkl.Layer):
    
    def __init__(self, 
                 encoder_layers,
                 image_shape=(256, 256, 1),
                 layer_filters=[1024, 1024, 512, 256, 128, 64],
                 name='decoder', **kwargs):
        """Transforms compressed vector representation back into a 2D image.
        Contains 6x 2D transposed convolutional layers, and filter weights are
        tied to a given encoder. 

        Args: 
            encoder_layers (list): List of 6 encoder convolutional layers from
                which weights will be shared. 
            image_shape (tuple, optional): Output image shape. Defaults to 
                (256, 256, 1). 
            layer_filters (list, optional): Number of filters in each 
                convolutional layer. Defaults to [1024, 1024, 512, 256, 128, 
                64]. 
            name (str, optional): Name. Defaults to 'decoder'.
        """        
        super(TiedDecoder, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape
        self.layer_filters = layer_filters
        
        tupReshape = (image_shape[0] // 64, image_shape[1] // 64, layer_filters[0])
        
        self.dense = tkl.Dense(np.product(tupReshape), name=name + '_dense')
        self.reshape = tkl.Reshape(tupReshape, name=name + '_reshape')
        self.prelu_dense = tkl.PReLU(name=name + '_prelu_dense')
                
        self.tconv0 = TiedConv2DTranspose(encoder_layers[5], layer_filters[1], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv0')
        self.bn0 = tkl.BatchNormalization(name=name+ '_bn0')
        self.prelu0 = tkl.PReLU(name=name + '_prelu0')
        
        self.tconv1 = TiedConv2DTranspose(encoder_layers[4], layer_filters[2], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv1')
        self.bn1 = tkl.BatchNormalization(name=name+ '_bn1')
        self.prelu1 = tkl.PReLU(name=name + '_prelu1')
        
        self.tconv2 = TiedConv2DTranspose(encoder_layers[3], layer_filters[3], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv2')
        self.bn2 = tkl.BatchNormalization(name=name+ '_bn2')
        self.prelu2 = tkl.PReLU(name=name + '_prelu2')
        
        self.tconv3 = TiedConv2DTranspose(encoder_layers[2], layer_filters[4], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv3')
        self.bn3 = tkl.BatchNormalization(name=name+ '_bn3')
        self.prelu3 = tkl.PReLU(name=name + '_prelu3')
        
        self.tconv4 = TiedConv2DTranspose(encoder_layers[1], layer_filters[5], 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv4')
        self.bn4 = tkl.BatchNormalization(name=name+ '_bn4')
        self.prelu4 = tkl.PReLU(name=name + '_prelu4')
        
        self.tconv5 = TiedConv2DTranspose(encoder_layers[0], 1, 4, 
                                          strides=(2, 2), padding='same', name=name + '_tconv5')
        self.bn5 = tkl.BatchNormalization(name=name+ '_bn5')
        self.sigmoid_out = tkl.Activation('sigmoid', name=name + '_sigmoid')
                
    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.prelu_dense(x)
        
        x = self.tconv0(x)
        x = self.bn0(x, training=training)
        x = self.prelu0(x)
        
        x = self.tconv1(x)
        x = self.bn1(x, training=training)
        x = self.prelu1(x)
        
        x = self.tconv2(x)
        x = self.bn2(x, training=training)
        x = self.prelu2(x)
        
        x = self.tconv3(x)
        x = self.bn3(x, training=training)
        x = self.prelu3(x)
        
        x = self.tconv4(x)
        x = self.bn4(x, training=training)
        x = self.prelu4(x)
        
        x = self.tconv5(x)
        x = self.bn5(x, training=training)
        x = self.sigmoid_out(x)
        
        return x
        
    def get_config(self):
        return {'image_shape': self.image_shape,
                'layer_filters': self.layer_filters}
        
class AuxClassifier(tkl.Layer):
           
    def __init__(self, 
                 units=32,
                 name='auxclassifier', **kwargs):
        """Simple dense classifier with one hidden layer and sigmoid output.

        Args:
            units (int, optional): Number of hidden layer neurons. Defaults to 32.
            name (str, optional): Name. Defaults to 'auxclassifier'.
        """        
        super(AuxClassifier, self).__init__(name=name, **kwargs)
        
        self.units = units
        
        self.hidden = tkl.Dense(units, name=name + '_dense')
        self.activation = tkl.LeakyReLU(name=name + '_leakyrelu')
        self.dense_out = tkl.Dense(1, activation='sigmoid', name=name + '_output')
        
    def call(self, inputs):
        x = self.hidden(inputs)
        x = self.activation(x)
        x = self.dense_out(x)
        return x
    
    def get_config(self):
        return {'units': self.units}
        
class AdversarialClassifier(tkl.Layer):
    
    def __init__(self, image_shape, n_clusters, 
                 layer_filters=[16, 32, 64, 128, 256, 512],
                 dense_units=256,
                 name='adversary', **kwargs):
        """Domain adversarial classifier whose inputs are the layer outputs 
        from a convolutional encoder with 6 layers.

        Args:
            image_shape (tuple): Original image shape.
            n_clusters (int): Number of possible clusters (domains), i.e. 
                the size of the softmax output.
            layer_filters (list, optional): Number of filters in 
                each adversary layer. Defaults to [16, 32, 64, 128, 256, 512].
            dense_units (int, optional): Number of neurons in 
                adversary dense layer. Defaults to 256.
            name (str, optional): Name. Defaults to 'adversary'.
        """        
        
        super(AdversarialClassifier, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape
        self.n_clusters = n_clusters
        self.layer_filters = layer_filters
        self.dense_units = dense_units
        
        def _get_conv_shape(depth):
            # Compute spatial dimensions of intermediate tensor
            h = image_shape[0] // (2 ** (depth + 1))
            w = image_shape[1] // (2 ** (depth + 1))
            return (h, w)
        
        self.conv0 = tkl.Conv2D(layer_filters[0], 4, strides=(2, 2), padding='same', name=name + '_conv0')
        self.bn0 = tkl.BatchNormalization(name=name + '_bn0')
        self.prelu0 = tkl.PReLU(name=name + '_prelu0')
        
        self.concat1 = tkl.Concatenate(axis=-1, name=name + '_concat1')
        self.conv1 = tkl.Conv2D(layer_filters[1], 4, strides=(2, 2), padding='same', name=name + '_conv1')
        self.bn1 = tkl.BatchNormalization(name=name + '_bn1')
        self.prelu1 = tkl.PReLU(name=name + '_prelu1')
        
        self.concat2 = tkl.Concatenate(axis=-1, name=name + '_concat2')
        self.conv2 = tkl.Conv2D(layer_filters[2], 4, strides=(2, 2), padding='same', name=name + '_conv2')
        self.bn2 = tkl.BatchNormalization(name=name + '_bn2')
        self.prelu2 = tkl.PReLU(name=name + '_prelu2')
        
        self.concat3 = tkl.Concatenate(axis=-1, name=name + '_concat3')
        self.conv3 = tkl.Conv2D(layer_filters[3], 4, strides=(2, 2), padding='same', name=name + '_conv3')
        self.bn3 = tkl.BatchNormalization(name=name + '_bn3')
        self.prelu3 = tkl.PReLU(name=name + '_prelu3')
        
        self.concat4 = tkl.Concatenate(axis=-1, name=name + '_concat4')
        self.conv4 = tkl.Conv2D(layer_filters[4], 4, strides=(2, 2), padding='same', name=name + '_conv4')
        self.bn4 = tkl.BatchNormalization(name=name + '_bn4')
        self.prelu4 = tkl.PReLU(name=name + '_prelu4')
        
        self.concat5 = tkl.Concatenate(axis=-1, name=name + '_concat5')
        self.conv5 = tkl.Conv2D(layer_filters[5], 4, strides=(2, 2), padding='same', name=name + '_conv5')
        self.bn5 = tkl.BatchNormalization(name=name + '_bn5')
        self.prelu5 = tkl.PReLU(name=name + '_prelu5')
        
        self.flatten = tkl.Flatten(name=name + '_flatten')
        self.concat_dense = tkl.Concatenate(axis=-1, name=name + '_concat_dense')
        self.dense = tkl.Dense(dense_units, name=name + '_dense')
        self.prelu_dense = tkl.PReLU(name=name + '_prelu_dense')
        
        self.dense_out = tkl.Dense(n_clusters, activation='softmax')
        
    def call(self, inputs):
        act0, act1, act2, act3, act4, act5, latents = inputs
        x = self.conv0(act0)
        x = self.bn0(x)
        x = self.prelu0(x)
        
        x = self.concat1([x, act1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        
        x = self.concat2([x, act2])
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        
        x = self.concat3([x, act3])
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        
        x = self.concat4([x, act4])
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.prelu4(x)
        
        x = self.concat5([x, act5])
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.prelu5(x)
        
        x = self.flatten(x)
        x = self.concat_dense([x, latents])
        x = self.dense(x)
        x = self.prelu_dense(x)
        x = self.dense_out(x)
        
        return x        
        
    def get_config(self):
        return {'image_shape': self.image_shape,
                'n_clusters': self.n_clusters,
                'layer_filters': self.layer_filters,
                'dense_units': self.dense_units}
        
        
class BaseAutoencoderClassifier(tf.keras.Model):
        
    def __init__(self,
                 image_shape=(256, 256, 1),
                 n_latent_dims=56, 
                 encoder_layer_filters=[64, 128, 256, 512, 1024, 1024],
                 classifier_hidden_units=32,
                 name='autoencoder',
                 **kwargs
                 ):
        """Basic autoencoder with auxiliary classifier to predict a binary 
        label from the latent representation.

        Args:
            image_shape (tuple, optional): Input image shape. Defaults to 
                (256, 256, 1).
            n_latent_dims (int, optional): Size of latent representation. 
                Defaults to 56.
            encoder_layer_filters (list, optional): Number of filters per 
                encoder layer. Defaults to [64, 128, 256, 512, 1024, 1024].
            classifier_hidden_units (int, optional): Number of hidden layer 
                neurons in the auxiliary classifier. Defaults to 32.
            name (str, optional): Name. Defaults to 'autoencoder'.
        """        
        
        super(BaseAutoencoderClassifier, self).__init__(name=name, **kwargs)
        
        self.image_shape = image_shape
        self.n_latent_dims = n_latent_dims
        self.encoder_layer_filters = encoder_layer_filters
        self.decoder_layer_filters = encoder_layer_filters[-1::-1]
        self.classifier_hidden_units = classifier_hidden_units
        
        self.encoder = Encoder(n_latent_dims=n_latent_dims,
                               layer_filters=encoder_layer_filters,
                               )
        
        lsEncoderLayers = [self.encoder.conv0,
                           self.encoder.conv1,
                           self.encoder.conv2,
                           self.encoder.conv3, 
                           self.encoder.conv4,
                           self.encoder.conv5]
        
        self.decoder = TiedDecoder(lsEncoderLayers, image_shape=image_shape, 
                                   layer_filters=self.decoder_layer_filters)
        
        self.classifier = AuxClassifier(units=classifier_hidden_units)
        
    def call(self, inputs, training=None):
        
        _, _, _, _, _, _, latent = self.encoder(inputs, training=training)
        recon = self.decoder(latent, training=training)
        classification = self.classifier(latent)
        
        return (recon, classification)
    
class DomainAdversarialAEC(BaseAutoencoderClassifier):
    
    def __init__(self,
                 image_shape=(256, 256, 1),
                 n_clusters=10,
                 n_latent_dims=56, 
                 encoder_layer_filters=[64, 128, 256, 512, 1024, 1024],
                 classifier_hidden_units=32,
                 adversary_layer_filters=[16, 32, 64, 128, 256, 512],
                 adversary_dense_units=256,
                 name='autoencoder',
                 **kwargs
                 ):
        """Domain adversarial autoencoder-classifier. Adds an adversarial 
        classifier to predict the cluster/domain membership of each sample 
        based on the encoder's intermediate outputs. This compels the 
        encoder to learn features unassociated with cluster characteristics.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to 
                (256, 256, 1).
            n_clusters (int, optional): Number of clusters. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation. 
                Defaults to 56.
            encoder_layer_filters (list, optional): Number of filters in 
                each encoder layer. Defaults to [64, 128, 256, 512, 1024, 1024].
            classifier_hidden_units (int, optional): Number of neurons in 
                auxiliary classifier hidden layer. Defaults to 32.
            adversary_layer_filters (list, optional): Number of filters in 
                each adversary layer. Defaults to [16, 32, 64, 128, 256, 512].
            adversary_dense_units (int, optional): Number of neurons in 
                adversary dense layer. Defaults to 256.
            name (str, optional): Model name. Defaults to 'autoencoder'.
        """        
        super(DomainAdversarialAEC, self).__init__(
            image_shape=image_shape,
            n_latent_dims=n_latent_dims,
            encoder_layer_filters=encoder_layer_filters,
            classifier_hidden_units=classifier_hidden_units,
            name=name, 
            **kwargs)
        
        self.adversary = AdversarialClassifier(image_shape, n_clusters,
                                               layer_filters=adversary_layer_filters,
                                               dense_units=adversary_dense_units)
        
    def call(self, inputs, training=None):
        images, clusters = inputs
        encoder_outs = self.encoder(images, training=training)
        latent = encoder_outs[-1]
        recon = self.decoder(latent, training=training)
        classification = self.classifier(latent)
        pred_cluster = self.adversary(encoder_outs)
        
        return (recon, classification, pred_cluster)
    
    def compile(self,
                loss_recon=tf.keras.losses.MeanSquaredError(),
                loss_class=tf.keras.losses.BinaryCrossentropy(),
                loss_adv=tf.keras.losses.BinaryCrossentropy(),
                metric_class=tf.keras.metrics.AUC(name='auroc'),
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                loss_recon_weight=1.0,
                loss_class_weight=0.01,
                loss_gen_weight=0.02,
                ):
        super().compile()

        self.loss_recon = loss_recon
        self.loss_class = loss_class
        self.loss_adv = loss_adv

        self.opt_autoencoder = opt_autoencoder
        self.opt_adversary = opt_adversary
        
        # Loss trackers
        self.loss_recon_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')
        self.loss_adv_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        self.metric_class = metric_class
        self.metric_adv = metric_adv
        
        self.loss_recon_weight = loss_recon_weight
        self.loss_class_weight = loss_class_weight
        self.loss_gen_weight = loss_gen_weight      
        
    @property
    def metrics(self):
        return [self.loss_recon_tracker,
                self.loss_class_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_class,
                self.metric_adv]
        
    def train_step(self, data):
        if len(data) == 3:
            (images, clusters), (_, labels), sample_weights = data
        else:
            (images, clusters), (_, labels) = data
            sample_weights = None
            
        encoder_outs = self.encoder(images, training=True)
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(encoder_outs)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        with tf.GradientTape(persistent=True) as gt2:
            pred_recon, pred_class, pred_cluster = self((images, clusters), training=True)
            loss_class = self.loss_class(labels, pred_class, sample_weight=sample_weights)
            loss_recon = self.loss_recon(images, pred_recon, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
            total_loss = (self.loss_recon_weight * loss_recon) \
                + (self.loss_class_weight * loss_class) \
                - (self.loss_gen_weight * loss_adv)
                
        lsWeights = self.encoder.trainable_variables + self.decoder.trainable_variables \
                + self.classifier.trainable_variables
        grads_aec = gt2.gradient(total_loss, lsWeights)
        self.opt_autoencoder.apply_gradients(zip(grads_aec, lsWeights))
        
        self.metric_class.update_state(labels, pred_class)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (images, clusters), (_, labels) = data
                        
        pred_recon, pred_class, pred_cluster = self((images, clusters), training=True)
        loss_class = self.loss_class(labels, pred_class)
        loss_recon = self.loss_recon(images, pred_recon)
        loss_adv = self.loss_adv(clusters, pred_cluster)
            
        total_loss = (self.loss_recon_weight * loss_recon) \
            + (self.loss_class_weight * loss_class) \
            - (self.loss_gen_weight * loss_adv)
                    
        self.metric_class.update_state(labels, pred_class)
        self.metric_adv.update_state(clusters, pred_cluster)
        
        self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
class ClusterScaleBiasBlock(tkl.Layer):
    
    def __init__(self,
                 n_features, 
                 post_loc_init_scale=0.25,
                 prior_scale=0.25,
                 kl_weight=0.001,
                 name='cluster', 
                 **kwargs):
        super(ClusterScaleBiasBlock, self).__init__(name=name, **kwargs)
        
        self.post_loc_init_scale = post_loc_init_scale
        self.prior_scale = prior_scale
        self.kl_weight = kl_weight
        
        self.instance_norm = InstanceNormalization(center=True, 
                                                   scale=True, 
                                                   name=name + '_instance_norm')
    
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
        self.multiply = tkl.Multiply(name=name + '_mult')
        self.add = tkl.Add(name=name + '_add')

    def call(self, inputs):
        x, z = inputs
        x = self.instance_norm(x)
        g = self.gammas(z)
        b = self.betas(z)
        m = self.multiply((x, g))
        s = self.add((m, b))
        
        return s
    
    def get_config(self):
        return {'post_loc_init_scale': self.post_loc_init_scale,
                'prior_scale': self.prior_scale,
                'kl_weight': self.kl_weight}       
  

# class RandomEffectsTransformer(tkl.Layer):
#     def __init__(self, 
#                  contract_layer_filters=[32, 64, 128],
#                  middle_layer_filters=[128, 128],
#                  expand_layer_filters=[64, 32, 1],
#                  post_loc_init_scale=0.1,
#                  prior_scale=0.1,
#                  kl_weight=0.001,
#                  name='transformer', **kwargs):
#         """
#         Args: 
#             n_latent_dims (int, optional): Size of compressed representation
#                 output. Defaults to 56. 
#             layer_filters (list, optional): Filters per convolutional layer. 
#                 Defaults to [64, 128, 256, 512, 1024, 1024].
#             name (str, optional): Name. Defaults to 'encoder'.
#         """        
#         super(RandomEffectsTransformer, self).__init__(name=name, **kwargs)
        
#         assert len(contract_layer_filters) == len(expand_layer_filters)
        
#         self.contract_layer_filters = contract_layer_filters
#         self.middle_layer_filters = middle_layer_filters
#         self.expand_layer_filters = expand_layer_filters
        
#         self.contract_blocks = []
#         for iLayer, nFilters in enumerate(contract_layer_filters):
#             self.contract_blocks += [(tkl.Conv2D(nFilters, 4, 
#                                                  strides=(2, 2), 
#                                                  padding='same',
#                                                  name=name + '_strideconv' + str(iLayer)),
#                                       tkl.PReLU(name=name + '_strideconv_prelu' + str(iLayer)))]
            
#         self.residual_blocks = []
#         for iLayer, nFilters in enumerate(middle_layer_filters):
#             self.residual_blocks += [(tkl.Conv2D(nFilters, 4,
#                                                  padding='same',
#                                                  name=name + '_conv' + str(iLayer)),
#                                       ClusterScaleBiasBlock(nFilters,
#                                                             post_loc_init_scale,
#                                                             prior_scale,
#                                                             kl_weight,
#                                                             name=name + '_conv_re' + str(iLayer)),
#                                       tkl.PReLU(name=name + '_conv_prelu' + str(iLayer)))]
            
#         self.expand_blocks = []            
#         for iLayer, nFilters in enumerate(expand_layer_filters):
#             if iLayer == (len(expand_layer_filters) - 1):
#                 activation = tkl.Activation('sigmoid', name=name+ '_sigmoid_out')
#             else:
#                 activation = tkl.PReLU(name=name + '_tconv_prelu' + str(iLayer))
            
#             # self.expand_blocks += [(tkl.Conv2DTranspose(nFilters, 4,
#             #                                             strides=(2, 2),
#             #                                             padding='same',
#             #                                             name=name + '_tconv' + str(iLayer)),
#             #                         activation)]
                                           
#             conv_source = self.contract_blocks[-iLayer - 1][0]
#             self.expand_blocks += [(TiedConv2DTranspose(conv_source,
#                                                         nFilters,
#                                                         4,
#                                                         strides=(2, 2),
#                                                         padding='same',
#                                                         name=name + '_tconv' + str(iLayer)),
#                                     activation)]
            
#     def call(self, inputs):
#         x, z = inputs
        
#         for conv, act in self.contract_blocks:
#             x = conv(x)
#             x = act(x)
            
#         for conv, re, act in self.residual_blocks:
#             x2 = conv(x)
#             x2 = re((x2, z))
#             x2 = act(x2)
#             x = x2 + x
            
#         for conv, act in self.expand_blocks:
#             x = conv(x)
#             x = act(x)
            
#         return x
        
#     def get_config(self):
#         return {'contract_layer_filters': self.contract_layer_filters,
#                 'middle_layer_filters': self.middle_layer_filters,
#                 'expand_layer_filters': self.expand_layer_filters}
    

class RandomEffectsTransformer(tkl.Layer):
    def __init__(self, 
                 n_clusters,
                 contract_layer_filters=[32, 64, 128],
                 expand_layer_filters=[64, 32, 1],
                 classifier_layer_filters=[16, 32, 64],
                 post_loc_init_scale=0.1,
                 prior_scale=0.1,
                 kl_weight=0.001,
                 name='transformer', **kwargs):
        """
        """        
        super(RandomEffectsTransformer, self).__init__(name=name, **kwargs)
        
        assert len(contract_layer_filters) == len(expand_layer_filters)
        
        self.n_clusters = n_clusters
        self.contract_layer_filters = contract_layer_filters
        self.expand_layer_filters = expand_layer_filters
        self.classifier_layer_filters = classifier_layer_filters
        
        self.contract_blocks = []
        for iLayer, nFilters in enumerate(contract_layer_filters):
            self.contract_blocks += [(tkl.Conv2D(nFilters, 4, 
                                                 strides=(2, 2), 
                                                 padding='same',
                                                 name=name + '_conv' + str(iLayer)),
                                      ClusterScaleBiasBlock(nFilters,
                                                            post_loc_init_scale,
                                                            prior_scale,
                                                            kl_weight,
                                                            name=name + '_conv_re' + str(iLayer)),
                                      tkl.PReLU(name=name + '_conv_prelu' + str(iLayer)))]
                                                  
        self.expand_blocks = []            
        for iLayer, nFilters in enumerate(expand_layer_filters):
            if iLayer == (len(expand_layer_filters) - 1):
                activation = tkl.Activation('sigmoid', name=name+ '_sigmoid_out')
            else:
                activation = tkl.PReLU(name=name + '_tconv_prelu' + str(iLayer))
                                                   
            conv_source = self.contract_blocks[-iLayer - 1][0]
            self.expand_blocks += [(TiedConv2DTranspose(conv_source,
                                                        nFilters,
                                                        4,
                                                        strides=(2, 2),
                                                        padding='same',
                                                        name=name + '_tconv' + str(iLayer)),
                                    ClusterScaleBiasBlock(nFilters,
                                                          post_loc_init_scale,
                                                          prior_scale,
                                                          kl_weight,
                                                          name=name + '_tconv_re' + str(iLayer)),
                                    activation)]
            
        self.classifier_blocks = []
        for iLayer, nFilters in enumerate(classifier_layer_filters):
            if iLayer == (len(classifier_layer_filters) - 1):
                activation = tkl.Activation('sigmoid', name=name+ '_clf_sigmoid_out')
            else:
                activation = tkl.PReLU(name=name + '_clf_conv_prelu' + str(iLayer))
            self.classifier_blocks += [(tkl.Conv2D(nFilters, 4,
                                                   strides=(2, 2),
                                                   padding='same',
                                                   name=name + '_clf_conv' + str(iLayer)),
                                       activation)]
            
        self.classifier_flatten = tkl.Flatten(name=name + '_clf_flatten')
        self.classifier_dense = tkl.Dense(512, name=name + '_clf_dense')
        self.classifier_out = tkl.Dense(n_clusters, activation='softmax', name=name+ '_clf_out')
            
    def call(self, inputs):
        x, z = inputs
        
        lsFeatureMaps = []
        for conv, re, act in self.contract_blocks:
            x = conv(x)
            x = re((x, z))
            x = act(x)
            lsFeatureMaps += [x]
                        
        for conv, re, act in self.expand_blocks:
            x = conv(x)
            x = re((x, z))
            x = act(x)
        
        for iLayer in range(len(self.classifier_blocks)):
            if iLayer == 0:
                c = lsFeatureMaps[0]
            else:
                c = tf.concat([c, lsFeatureMaps[iLayer]], axis=-1)
            
            conv, act = self.classifier_blocks[iLayer]
            c = conv(c)
            c = act(c)
                         
        c = self.classifier_flatten(c)
        c = self.classifier_dense(c)
        c = self.classifier_out(c)
                         
        return x, c
        
    def get_config(self):
        return {'contract_layer_filters': self.contract_layer_filters,
                'expand_layer_filters': self.expand_layer_filters,
                'n_clusters': self.n_clusters}
    
class MixedEffectsAEC(DomainAdversarialAEC):
    
    def __init__(self,
                 image_shape=(256, 256, 1),
                 n_clusters=10,
                 n_latent_dims=56, 
                 encoder_layer_filters=[64, 128, 256, 512, 1024, 1024],
                 classifier_hidden_units=32,
                 contract_layer_filters=[32, 64, 128],
                 middle_layer_filters=[128, 128],
                 expand_layer_filters=[64, 32, 1],
                 re_classifier_layer_filters=[16, 32, 64],
                 post_loc_init_scale=0.1,
                 prior_scale=0.1,
                 kl_weight=0.001,
                 name='autoencoder',
                 **kwargs
                 ):
        """Mixed effects autoencoder, including a second autoencoder to learn
        cluster-specific random effects that are separated from the main 
        autoencoder.

        Args:
            image_shape (tuple, optional): Image shape. Defaults to 
                (256, 256, 1).
            n_clusters (int, optional): Number of clusters. Defaults to 10.
            n_latent_dims (int, optional): Size of latent representation. 
                Defaults to 56.
            encoder_layer_filters (list, optional): Number of filters in 
                each encoder layer. Defaults to [64, 128, 256, 512, 1024, 1024].
            classifier_hidden_units (int, optional): Number of neurons in 
                auxiliary classifier hidden layer. Defaults to 32.
            name (str, optional): Model name. Defaults to 'autoencoder'.
        """        
        super(MixedEffectsAEC, self).__init__(
            image_shape=image_shape,
            n_clusters=n_clusters,
            n_latent_dims=n_latent_dims,
            encoder_layer_filters=encoder_layer_filters,
            classifier_hidden_units=classifier_hidden_units,
            name=name, 
            **kwargs)
        
        self.transformer = RandomEffectsTransformer(n_clusters,
            contract_layer_filters=contract_layer_filters,
            # middle_layer_filters=middle_layer_filters,
            expand_layer_filters=expand_layer_filters,
            classifier_layer_filters=re_classifier_layer_filters,
            post_loc_init_scale=post_loc_init_scale,
            prior_scale=prior_scale,
            kl_weight=kl_weight)
        
    def call(self, inputs, training=None):
        images, clusters = inputs
        
        recon_fe, classification, pred_cluster = super(MixedEffectsAEC, self).call(inputs, 
                                                                                   training=training)
        recon_me, pred_cluster_re = self.transformer((recon_fe, clusters))
                
        return (recon_me, recon_fe, classification, pred_cluster, pred_cluster_re)
    
    def compile(self,
                loss_recon=tf.keras.losses.MeanSquaredError(),
                loss_class=tf.keras.losses.BinaryCrossentropy(),
                loss_adv=tf.keras.losses.CategoricalCrossentropy(),
                metric_class=tf.keras.metrics.AUC(name='auroc'),
                metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                metric_re_cluster=tf.keras.metrics.CategoricalAccuracy(name='re_acc'),
                opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                loss_recon_me_weight=1.0,
                loss_recon_fe_weight=1.0,
                loss_class_weight=0.01,
                loss_gen_weight=0.02,
                loss_re_cluster_weight=0.1,
                ):
        super(MixedEffectsAEC, self).compile(
            loss_recon=loss_recon,
            loss_class=loss_class,
            loss_adv=loss_adv,
            metric_class=metric_class,
            metric_adv=metric_adv,
            opt_autoencoder=opt_autoencoder,
            opt_adversary=opt_adversary,
            loss_recon_weight=0,
            loss_class_weight=loss_class_weight,
            loss_gen_weight=loss_gen_weight)
        
        # Loss trackers
        del self.loss_recon_tracker, self.loss_recon_weight
        
        self.loss_recon_me_weight = loss_recon_me_weight
        self.loss_recon_fe_weight = loss_recon_fe_weight
        self.loss_recon_me_tracker = tf.keras.metrics.Mean(name='recon_me_loss')
        self.loss_recon_fe_tracker = tf.keras.metrics.Mean(name='recon_fe_loss')
                
        self.metric_re_cluster = metric_re_cluster
        self.loss_re_cluster_weight = loss_re_cluster_weight
        
    @property
    def metrics(self):
        return [self.loss_recon_me_tracker,
                self.loss_recon_fe_tracker,
                self.loss_class_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_class,
                self.metric_adv,
                self.metric_re_cluster]
        
    def train_step(self, data):
        if len(data) == 3:
            (images, clusters), (_, labels), sample_weights = data
        else:
            (images, clusters), (_, labels) = data
            sample_weights = None
            
        encoder_outs = self.encoder(images, training=True)
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(encoder_outs)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)
        
        with tf.GradientTape(persistent=True) as gt2:
            pred_recon_me, pred_recon_fe, pred_class, pred_cluster, pred_cluster_re = \
                self((images, clusters), training=True)
            loss_class = self.loss_class(labels, pred_class, sample_weight=sample_weights)
            loss_recon_me = self.loss_recon(images, pred_recon_me, sample_weight=sample_weights)
            loss_recon_fe = self.loss_recon(images, pred_recon_fe, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            loss_cluster_re = self.loss_adv(clusters, pred_cluster_re, sample_weight=sample_weights)
            
            total_loss = (self.loss_recon_me_weight * loss_recon_me) \
                + (self.loss_recon_fe_weight * loss_recon_fe) \
                + (self.loss_class_weight * loss_class) \
                - (self.loss_gen_weight * loss_adv) \
                + (self.loss_re_cluster_weight * loss_cluster_re) \
                + self.transformer.losses
                
        lsWeights = self.encoder.trainable_variables \
                    + self.decoder.trainable_variables \
                    + self.classifier.trainable_variables \
                    + self.transformer.trainable_variables
        grads_aec = gt2.gradient(total_loss, lsWeights)
        self.opt_autoencoder.apply_gradients(zip(grads_aec, lsWeights))
        
        self.metric_class.update_state(labels, pred_class)
        self.metric_re_cluster.update_state(clusters, pred_cluster_re)
        self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_me_tracker.update_state(loss_recon_me)
        self.loss_recon_fe_tracker.update_state(loss_recon_fe)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (images, clusters), (_, labels) = data
                        
        pred_recon_me, pred_recon_fe, pred_class, pred_cluster, pred_cluster_re = \
            self((images, clusters), training=True)
        loss_class = self.loss_class(labels, pred_class)
        loss_recon_me = self.loss_recon(images, pred_recon_me)
        loss_recon_fe = self.loss_recon(images, pred_recon_fe)
        loss_adv = self.loss_adv(clusters, pred_cluster)
        loss_cluster_re = self.loss_adv(clusters, pred_cluster_re)
            
        total_loss = (self.loss_recon_me_weight * loss_recon_me) \
            + (self.loss_recon_fe_weight * loss_recon_fe) \
            + (self.loss_class_weight * loss_class) \
            - (self.loss_gen_weight * loss_adv) \
            + (self.loss_re_cluster_weight * loss_cluster_re) \
                    
        self.metric_class.update_state(labels, pred_class)
        self.metric_adv.update_state(clusters, pred_cluster)
        self.metric_re_cluster.update_state(clusters, pred_cluster_re)
        
        self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_me_tracker.update_state(loss_recon_me)
        self.loss_recon_fe_tracker.update_state(loss_recon_fe)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}