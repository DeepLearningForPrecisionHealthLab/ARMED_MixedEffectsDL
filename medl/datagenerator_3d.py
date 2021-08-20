'''
Custom Keras data generator classes for 3D images with random effect grouping.
'''

import os
import re
import numpy as np
import glob
import PIL
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K

class SegmentationDataFrameGenerator3D(Sequence):
    def __init__(self, dataframe, image_shape, 
                 mask_shape=None, 
                 return_group=False,
                 group_encoding=None,
                 batch_size=32, 
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 min_max_scale=False,
                 augmentation=None, 
                 channel=None,
                 shuffle=True,
                 seed=None):
        """Data generator class which takes a dataframe containing paths to images and labels and group memberships. 
        Iterating over this generator returns batches of (input, group design matrix), labels

        Args:
            dataframe (pandas.DataFrame): dataframe with columns [image, mask, group]. Columns `image` and `mask` should contain paths to .npy files.
            image_shape (tup): shape of images to output
            mask_shape (tup): shape of mask images. If none is specified, assumes that masks have shape image_shape[:3] + (1,) (single label). Defaults to None.
            return_group (bool): Whether to return the group membership of each image as an additional array. Set to False if training a non-ME model. Defaults to False.
            group_encoding (list): List containing the order of group names for conversion to one-hot encoding. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 32.
            samplewise_center (bool, optional): Normalize each sample to zero mean. Defaults to False.
            samplewise_std_normalization (bool, optional): Normalize each sample to unit s.d. Defaults to False.
            min_max_scale (bool, optional): Scale each sample to [0, 1], cannot be used with samplewise_center and samplewise_std_normalization. Defaults to None.
            channel (int, optional): Keep only this modality/channel of the multi-modality MRI image. E.g. 0 for T1, 1 for T1GD, 2 for T2, 3 for FLAIR.
            shuffle (bool, optional): Shuffle data after each epoch. Defaults to True.
            seed (int, optional): Random seed for shuffling data.

        Raises:
            ValueError: [description]
        """        

        self.batch_size = batch_size
        self.dataframe = dataframe
        self.image_shape = image_shape
        if mask_shape:
            self.mask_shape = mask_shape
        else:
            self.mask_shape = image_shape[:3] + (1,)
        self.return_group = return_group
        self.channel = channel
        self.shuffle = shuffle
        self.n_total_epochs_seen = -1
        self.seed = seed

        if min_max_scale & (samplewise_center | samplewise_std_normalization):
            raise ValueError('min_max_scale cannot be used if samplewise_center or samplewise_std_normalization are True')
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.min_max_scale = min_max_scale

        self.images = self.dataframe['image'].values
        self.masks = self.dataframe['mask'].values
        if group_encoding is not None:
            self.group_encoder = OneHotEncoder(categories=[group_encoding], drop=None, sparse=False, 
                                               handle_unknown='ignore',
                                               dtype=K.floatx())
        else:
            self.group_encoder = OneHotEncoder(drop=None, sparse=False, dtype=K.floatx(), handle_unknown='ignore')
        arrGroups = self.dataframe['group'].values.reshape((-1, 1))
        self.groups = self.group_encoder.fit_transform(arrGroups)
        
        if shuffle:
            self.on_epoch_end() # shuffle samples
        else:
            self.index = np.arange(len(self.images))

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, batch_index):
        batch_end = np.min([((batch_index + 1) * self.batch_size), self.index.shape[0]])
        samples = self.index[(batch_index * self.batch_size):batch_end]
        return self.__get_data(samples)

    def __get_data(self, samples):
        arrBatchX = np.zeros((samples.shape[0],) + self.image_shape)
        arrBatchY = np.zeros((samples.shape[0],) + self.mask_shape)
        arrBatchGroup = np.zeros((samples.shape[0],) + (self.groups.shape[1],))
        for i, idx in enumerate(samples):
            img = np.load(self.images[idx])
            if self.channel:
                img = img[..., self.channel:(self.channel + 1)]
            mask = np.load(self.masks[idx])
            if len(img.shape) < 4:
                img = np.expand_dims(img, axis=-1)
            if len(mask.shape) < 4:
                mask = np.expand_dims(mask, axis=-1)

            if img.shape[:3] != self.image_shape[:3]:
                raise IOError('Images do not conform to the specified shape of {}. Resizing on-the-fly is not supported yet.'.format(self.image_shape))

            img = img.astype(K.floatx())
            mask = mask.astype(K.floatx())

            if self.samplewise_center:
                img -= img.mean()
            if self.samplewise_std_normalization:
                img /= img.std()
            if self.min_max_scale:
                img -= img.min()
                img /= img.max()

            arrBatchX[i,] = img
            arrBatchY[i,] = mask > 0
            arrBatchGroup[i,] = self.groups[idx]
        if self.return_group:
            return (arrBatchX, arrBatchGroup), arrBatchY
        else:
            return arrBatchX, arrBatchY

    def on_epoch_end(self):
        self.index = np.arange(len(self.images))
        if self.seed:
            np.random.seed(self.seed + self.n_total_epochs_seen)
        if self.shuffle:
            np.random.shuffle(self.index)
        self.n_total_epochs_seen += 1
        return super().on_epoch_end()