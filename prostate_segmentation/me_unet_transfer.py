'''
Transfer trained weights from the base Unet model, add random effects layers, then fine tune to learn the random effects.
'''

DATADIR = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_me_20210110/random_slope_featurewise_transfer'
WEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_20210110/test/final_weights.h5' # weights for base Unet
SEED = 493
IMAGESHAPE = (384, 384, 1)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
sys.path.append('../')
from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
import albumentations
from prostate_datagenerator import SegmentationDataGenerator # pylint: disable=import-error
from random_effects_prototype import DenseRandomEffects

strOutputDir = os.path.join(RESULTSDIR, 'val')
os.makedirs(strOutputDir, exist_ok=True)
  
# Define augmentation pipeline
# pylint: disable=no-member
augmentations = albumentations.Compose([albumentations.VerticalFlip(p=0.5),
                                        albumentations.HorizontalFlip(p=0.5),
                                        albumentations.ShiftScaleRotate(
                                            shift_limit=0.1, 
                                            scale_limit=0.1, 
                                            rotate_limit=20, 
                                            interpolation=cv2.INTER_CUBIC,
                                            border_mode=cv2.BORDER_CONSTANT,
                                            p=0.5)
                                        ])                                               

# Create data generators
train_data = SegmentationDataGenerator(os.path.join(DATADIR, 'train', 'image'),
                                        os.path.join(DATADIR, 'train', 'mask'),
                                        IMAGESHAPE,
                                        batch_size=10,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        augmentation=augmentations,
                                        return_group=True,
                                        seed=SEED)
val_data = SegmentationDataGenerator(os.path.join(DATADIR, 'val', 'image'),
                                        os.path.join(DATADIR, 'val', 'mask'),
                                        IMAGESHAPE,
                                        batch_size=10,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        return_group=True,
                                        seed=SEED)

# Create base Unet with pretrained weights
model = unet.unet(input_size=IMAGESHAPE, pretrained_weights=WEIGHTS)
# Add random effect slope to middle of model. The following code to insert 
# an intermediate layer was borrowed from 
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
inputsRE = tf.keras.layers.Input((5,))
reSlope = DenseRandomEffects(1024, 
                            loc_init_range=1, 
                            scale_init_range=(0.1, 0.2),
                            prior_scale=0.1)(inputsRE) 
# Auxiliary dictionary to describe the network graph
network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

# Set the input layers of each layer
for layer in model.layers:
    for node in layer._outbound_nodes:
        layer_name = node.outbound_layer.name
        if layer_name not in network_dict['input_layers_of']:
            network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
        else:
            network_dict['input_layers_of'][layer_name].append(layer.name)

# Set the output tensor of the input layer
network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

# Iterate over all layers after the input
model_outputs = []
for layer in model.layers[1:]:

    # Determine input tensors
    layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
            for layer_aux in network_dict['input_layers_of'][layer.name]]
    if len(layer_input) == 1:
        layer_input = layer_input[0]

    # Insert layer if name matches the regular expression
    if layer.name == 'conv2d_9':
        x = layer(layer_input)

        re = tf.keras.layers.Multiply()([x, reSlope]) 
        x = tf.keras.layers.Add()([re, x])
        print('added layer after', layer.name)
    else:
        x = layer(layer_input)

    # Set new output tensor (the original one, or the one of the inserted
    # layer)
    network_dict['new_output_tensor_of'].update({layer.name: x})

    # Save tensor in output list if it is output in initial model
    if layer.name in model.output_names:
        model_outputs.append(x)

modelRE = tf.keras.models.Model(model.inputs + [inputsRE], model_outputs)
modelRE.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
            loss=dice_bce_loss, 
            metrics=[dice])

strLogDir = os.path.join(strOutputDir, 'logs')
os.makedirs(strLogDir, exist_ok=True)
lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max'),
                tf.keras.callbacks.CSVLogger(os.path.join(strOutputDir, 'training.log')),
                SaveImagesCallback(val_data[0][0], val_data[0][1], strOutputDir, random_effects=True),
                tf.keras.callbacks.TensorBoard(log_dir=strLogDir)]

modelRE.fit(train_data,
        validation_data=val_data,
        epochs=200,
        batch_size=32,
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data),
        callbacks=lsCallbacks,
        verbose=1)
modelRE.save_weights(os.path.join(strOutputDir, 'final_weights.h5'))    