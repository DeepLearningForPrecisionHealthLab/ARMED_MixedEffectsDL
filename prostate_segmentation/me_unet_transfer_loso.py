'''
Transfer trained weights from the base Unet model, add random effects layers, then fine tune to learn the random effects.
'''

SPLITS = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/slices/LOSO_splits.pkl'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_LOSO/me_unet'
WEIGHTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_LOSO/base_unet'
SEED = 493
IMAGESHAPE = (384, 384, 1)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import argparse
import pickle
import tensorflow as tf
import numpy as np
import cv2
sys.path.append('../')
from medl.models import unet # pylint: disable=import-error
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
import albumentations
from prostate_datagenerator import SegmentationDataFrameGenerator # pylint: disable=import-error
from random_effects_prototype import DenseRandomEffects

def train(strSite):
    strOutputDir = os.path.join(RESULTSDIR, strSite)
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
    with open(SPLITS, 'rb') as f:
        dictSplits = pickle.load(f)
    lsGroups = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI15', 'UCL']
    lsGroups.remove(strSite)
    train_data = SegmentationDataFrameGenerator(dictSplits[strSite]['train'],
                                                IMAGESHAPE,
                                                return_group=True,
                                                group_encoding=lsGroups,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                augmentation=augmentations,
                                                seed=SEED)
    val_data = SegmentationDataFrameGenerator(dictSplits[strSite]['test'],
                                                IMAGESHAPE,
                                                return_group=True,
                                                group_encoding=lsGroups,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)
    test_data = SegmentationDataFrameGenerator(dictSplits[strSite]['test'],
                                                IMAGESHAPE,
                                                return_group=True,
                                                group_encoding=lsGroups,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)
    test_data_nore = SegmentationDataFrameGenerator(dictSplits[strSite]['test'],
                                                IMAGESHAPE,
                                                return_group=True,
                                                group_encoding=lsGroups,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)                                                
    loso_data = SegmentationDataFrameGenerator(dictSplits[strSite]['held-out site'],
                                                IMAGESHAPE,
                                                return_group=True,
                                                batch_size=16,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                seed=SEED)  
    # Create dummy design matrix
    test_data_nore.groups = np.zeros((len(test_data_nore.images), 5))
    loso_data.groups = np.zeros((len(loso_data.images), 5))

    # Create base Unet with pretrained weights
    strWeightsPath = os.path.join(WEIGHTSDIR, strSite, 'final_weights.h5')
    model = unet.unet(input_size=IMAGESHAPE, pretrained_weights=strWeightsPath)
    # Add random effect slope to middle of model. The following code to insert 
    # an intermediate layer was borrowed from 
    # https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
    inputsRE = tf.keras.layers.Input((5,))
    reSlope = DenseRandomEffects(1024, 
                                loc_init_range=0.1, 
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
            steps_per_epoch=len(train_data),
            validation_steps=len(val_data),
            callbacks=lsCallbacks,
            verbose=1)
    modelRE.save_weights(os.path.join(strOutputDir, 'final_weights.h5'))  

    fDiceTrain = modelRE.evaluate(train_data)[1]
    fDiceTest = modelRE.evaluate(test_data)[1]
    fDiceTestNoRE = modelRE.evaluate(test_data_nore)[1]
    fDiceLOSO = modelRE.evaluate(loso_data)[1]

    with open(os.path.join(strOutputDir, 'results.txt'), 'w') as f:
        f.write(f'Train: {fDiceTrain}\nTest: {fDiceTest}\nTest w/o RE: {fDiceTestNoRE}\n{strSite}: {fDiceLOSO}')
  

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--site', '-s', type=str, help='Site to hold out of training')
    arguments = args.parse_args()
    train(arguments.site)