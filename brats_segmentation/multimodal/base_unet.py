'''
Train classic U-Net to segment GBMs from multimodal MRI axial slices.
'''
DATADIR = '/archive/bioinformatics/DLLab/KevinNguyen/data/BraTS2020'
RESULTSDIR = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/gbm_segmentation_20210104'
SEED = 399
FOLD = 0

import os
import sys
import tensorflow as tf
sys.path.append('../../')
from medl import unet # pylint: disable=import-error
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = tf.config.experimental.list_physical_devices()
print(gpus)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Dice coefficient metric and loss
def dice(yTrue, yPred):
    # Computes the hard dice score
    yPredBinary = tf.keras.backend.cast_to_floatx(yPred >= 0.5)
    intersection = tf.reduce_sum(tf.multiply(yTrue, yPredBinary), axis=(1, 2, 3))
    total = tf.reduce_sum(yTrue, axis=(1, 2, 3)) + tf.reduce_sum(yPredBinary, axis=(1, 2, 3))
    dicecoef = tf.reduce_mean((2.0 * intersection + 1.0) / (total + 1.0))
    return dicecoef

def dice_loss(yTrue, yPred):
    # Sum of the soft dice loss and BCE loss (dice alone doesn't seem to train well)
    intersection = tf.reduce_sum(tf.multiply(yTrue, yPred), axis=(1, 2, 3))
    total = tf.reduce_sum(yTrue, axis=(1, 2, 3)) + tf.reduce_sum(yPred, axis=(1, 2, 3))
    dicecoef = tf.reduce_mean((2.0 * intersection + 1.0) / (total + 1.0))
    return 1 - dicecoef + tf.keras.losses.binary_crossentropy(yTrue, yPred)

# # Image preprocessing function, normalizes values to [0, 1]
# def preprocess(arrImage):
#     return arrImage / arrImage.max(axis=(0, 1))[np.newaxis, np.newaxis, :]

class SaveImagesCallback(tf.keras.callbacks.Callback):
    def  __init__(self, arrXVal, arrYVal, strSaveDir):
        ''' Callback for saving some example segmentations as images at the end of each epoch
        '''
        super(SaveImagesCallback, self).__init__()
        self.arrXVal = arrXVal
        self.arrYVal = arrYVal
        self.strSaveDir = strSaveDir
    def on_epoch_end(self, epoch, logs=None):
        fig, ax = plt.subplots(4, 4, dpi=150)
        # Create a grid where each row is a different sample and each column 
        # is a different modality within that sample
        for i in range(4):
            np.random.seed(i * 2348)
            k = np.random.randint(self.arrXVal.shape[0])
            arrInput = self.arrXVal[k,] 
            arrTrueMask = self.arrYVal[k,].squeeze()
            arrPredMask = self.model.predict(np.expand_dims(arrInput, 0)).squeeze()
            for j in range(4):
                ax[i, j].imshow(arrInput[:, :, j], cmap='Greys_r')

                arrTrueOverlay = np.zeros(arrTrueMask.shape + (4,))
                arrTrueOverlay[..., 0] = arrTrueMask
                arrTrueOverlay[..., -1] = arrTrueMask
                ax[i, j].imshow(arrTrueOverlay, alpha=0.3)

                arrPredOverlay = np.zeros(arrTrueMask.shape + (4,))
                arrPredOverlay[..., 2] = (arrPredMask >= 0.5)
                arrPredOverlay[..., -1] = (arrPredMask)
                ax[i, j].imshow(arrPredOverlay, alpha=0.3)
                ax[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.strSaveDir, 'epoch{:03d}.png'.format(epoch)))

# Load cross-validation data
strDataPath = os.path.join(DATADIR, f'fold{FOLD}.npz')
dictData = np.load(strDataPath)
arrTrainData = np.concatenate((dictData['t1_train'], dictData['t1ce_train'], dictData['t2_train'], dictData['flair_train']), axis=-1)
arrTrainLabel = dictData['mask_train']
arrValData = np.concatenate((dictData['t1_val'], dictData['t1ce_val'], dictData['t2_val'], dictData['flair_val']), axis=-1)
arrValLabel = dictData['mask_val']
tupImgShape = arrTrainData.shape[1:]
print('Images are size', tupImgShape)
del dictData

# Create data generators
# train_data = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,
#                                                              width_shift_range=0.2,
#                                                              height_shift_range=0.2,
#                                                              shear_range=0.2,
#                                                              zoom_range=0.1)
#TODO: use albumentations instead of Keras's augmentation. Keras does not transform the label mask to match the input!!!
# train_data = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,
#                                                              width_shift_range=0.2,
#                                                              height_shift_range=0.2,
#                                                              shear_range=0.2,
#                                                              zoom_range=0.1)
# train_label = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
#                                                              height_shift_range=0.2,
#                                                              shear_range=0.2,
#                                                              zoom_range=0.1)
# val_data = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,
#                                                              width_shift_range=0.2,
#                                                              height_shift_range=0.2,
#                                                              shear_range=0.2,
#                                                              zoom_range=0.1)
# val_label = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
#                                                              height_shift_range=0.2,
#                                                              shear_range=0.2,
#                                                              zoom_range=0.1)
# apparently setting the same random seed for both the input and label generators should match up the transformations
# this needs to be double-checked though
# train_generator = zip(train_data.flow(arrTrainData.astype(np.float32), seed=SEED),
#                       train_label.flow(arrTrainLabel.astype(np.float32), seed=SEED))
# val_generator = zip(val_data.flow(arrValData.astype(np.float32), seed=SEED),
#                     val_label.flow(arrValLabel.astype(np.float32), seed=SEED)) 
# 
train_data = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
val_data = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)        

model = unet.unet(input_size=tupImgShape)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4), 
              loss=dice_loss, 
              metrics = [dice])
lsCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice', patience=20, restore_best_weights=True, mode='max'),
              SaveImagesCallback(arrValData, arrValLabel, RESULTSDIR),
              tf.keras.callbacks.ModelCheckpoint(os.path.join(RESULTSDIR, 'weights.{epoch:03d}.hdf5'),
                                                monitor='val_dice',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                save_freq=5 * arrTrainData.shape[0] // 32),
              tf.keras.callbacks.CSVLogger(os.path.join(RESULTSDIR, 'training.log'))]
nTrainSamples = arrTrainData.shape[0]
nValSamples = arrValData.shape[0]
history = model.fit(train_data.flow(arrTrainData.astype(np.float32), arrTrainLabel.astype(np.float32)),
                    validation_data=val_data.flow(arrValData.astype(np.float32), arrValLabel.astype(np.float32)),
                    epochs=500,
                    batch_size=32,
                    callbacks=lsCallbacks,
                    verbose=1)
df = pd.DataFrame(history.history)
df.to_csv(os.path.join(RESULTSDIR, 'history.csv'))
