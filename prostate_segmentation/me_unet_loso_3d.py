'''
Train ME U-Net to segment the prostate from T2-weighted MRIs
'''
SPLITS = '/endosome/archive/bioinformatics/DLLab/KevinNguyen/data/multisite_prostate/3d/LOSO_splits.pkl'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/prostate_segmentation_3d_LOSO_20210210/me_unet'
SEED = 493
IMAGESHAPE = (64, 64, 32, 1)

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pickle
sys.path.append('../')
from medl.models.losses import dice_bce_loss, dice # pylint: disable=import-error
# from medl.models.callbacks import SaveImagesCallback # pylint: disable=import-error
from prostate_datagenerator import SegmentationDataFrameGenerator3D # pylint: disable=import-error
from candidate_architectures.random_slope_int_l1norm import me_unet3d

def train(strSite):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices()
    print(gpus)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    strOutputDir = os.path.join(RESULTSDIR, strSite)
    os.makedirs(strOutputDir, exist_ok=True)

    # Create data generators
    with open(SPLITS, 'rb') as f:
        dictSplits = pickle.load(f)
    # Batch size of 16 is too large for the ME model (though it was fine for the base model)
    train_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['train'],
                                                IMAGESHAPE,
                                                batch_size=12,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                seed=SEED)
    nGroups = train_data.group_encoder.categories_[0].size
    val_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['val'],
                                                IMAGESHAPE,
                                                batch_size=12,
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
                                                return_group=True,
                                                group_encoding=train_data.group_encoder.categories_[0],
                                                seed=SEED)
    # test_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['test'],
    #                                             IMAGESHAPE,
    #                                             batch_size=16,
    #                                             samplewise_center=True,
    #                                             samplewise_std_normalization=True,
    #                                             return_group=True,
    #                                             group_encoding=train_data.group_encoder.categories_[0],
    #                                             seed=SEED)                                                
    # loso_data = SegmentationDataFrameGenerator3D(dictSplits[strSite]['held-out site'],
    #                                             IMAGESHAPE,
    #                                             batch_size=16,
    #                                             samplewise_center=True,
    #                                             samplewise_std_normalization=True,
    #                                             return_group=True,
    #                                             group_encoding=train_data.group_encoder.categories_[0],
    #                                             seed=SEED)                                                

    
    model = me_unet3d(IMAGESHAPE, nGroups)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
                loss=dice_bce_loss, 
                metrics=[dice])

    strLogDir = os.path.join(strOutputDir, 'logs')
    os.makedirs(strLogDir, exist_ok=True)
    lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max'),
                   tf.keras.callbacks.CSVLogger(os.path.join(strOutputDir, 'training.log')),
                #    SaveImagesCallback(val_data[0][0], val_data[0][1], strOutputDir),
                   tf.keras.callbacks.TensorBoard(log_dir=strLogDir)]

    model.fit(train_data,
            validation_data=val_data,
            epochs=200,
            steps_per_epoch=len(train_data),
            validation_steps=len(val_data),
            callbacks=lsCallbacks,
            verbose=1)
    model.save_weights(os.path.join(strOutputDir, 'final_weights.h5'))                    

    fDiceTrain = model.evaluate(train_data)[1]
    fDiceVal = model.evaluate(val_data)[1]
    # fDiceTest = model.evaluate(test_data)[1]
    # fDiceLOSO = model.evaluate(loso_data)[1]

    with open(os.path.join(strOutputDir, 'results.txt'), 'w') as f:
        # f.write(f'Train: {fDiceTrain}\nVal: {fDiceVal}\nTest: {fDiceTest}\n{strSite}: {fDiceLOSO}')
        f.write(f'Train: {fDiceTrain}\nVal: {fDiceVal}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--site', '-s', type=str, help='Site to hold out of training')
    arguments = args.parse_args()
    train(arguments.site)