'''
Custom Keras-compatible data generators for data with associated group labels
'''
import os
import re
import numpy as np
import glob
import PIL
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K


class SegmentationDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, image_shape,
                 return_group=False,
                 group_encoding=None,
                 batch_size=32, 
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 min_max_scale=False,
                 augmentation=None, 
                 shuffle=True,
                 seed=None):
        """Data generator class for data with associated group labels. Supports augmentation with albumentation pipelines.
        Iterating over this generator returns batches of (input, groups), labels

        Args:
            image_dir (str): path to directory containing images
            mask_dir (str): path to directory containing binary masks
            image_shape (tup): shape of images to output
            return_group (bool): Whether to return the group membership of each image as an additional array
            group_encoding (list): List containing the order of group names for conversion to one-hot encoding
            batch_size (int, optional): Batch size. Defaults to 32.
            samplewise_center (bool, optional): Normalize each sample to zero mean. Defaults to False.
            samplewise_std_normalization (bool, optional): Normalize each sample to unit s.d. Defaults to False.
            min_max_scale (bool, optional): Scale each sample to [0, 1], cannot be used with samplewise_center and samplewise_std_normalization. Defaults to None.
            augmentation (albumentation pipeline, optional): Augmentation pipeline. Defaults to None.
            shuffle (bool, optional): Shuffle data after each epoch. Defaults to True.
            seeed (int, optional): Random seed for shuffling data.

        Raises:
            ValueError: [description]
        """        

        self.batch_size = batch_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_shape = image_shape
        self.return_group = return_group
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.n_total_epochs_seen = -1
        self.seed = seed

        if min_max_scale & (samplewise_center | samplewise_std_normalization):
            raise ValueError('min_max_scale cannot be used if samplewise_center or samplewise_std_normalization are True')
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.min_max_scale = min_max_scale

        # Determine image + mask pairings
        self.images = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        def _match_mask(img_path):
            strImgFile = os.path.basename(img_path)
            regmatch = re.search(r'(.+)_(.*)_slice(\d+).png', strImgFile)
            strSubject = regmatch[1]
            strGroup = regmatch[2]
            strSlice = regmatch[3]
            strMaskPath = os.path.join(self.mask_dir, f'{strSubject}_{strGroup}_slice{strSlice}.png')
            if not os.path.exists(strMaskPath):
                raise FileNotFoundError(strMaskPath)
            return strMaskPath, strGroup
        lsMasksGroups = [_match_mask(s) for s in self.images]
        self.masks = [t[0] for t in lsMasksGroups]
        if group_encoding:
            self.group_encoder = OneHotEncoder(categories=[group_encoding], drop=None, sparse=False, dtype=K.floatx())
        else:
            self.group_encoder = OneHotEncoder(drop=None, sparse=False, dtype=K.floatx())
        arrGroups = np.array([t[1] for t in lsMasksGroups], dtype='object').reshape((-1, 1))
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
        arrBatchY = np.zeros((samples.shape[0],) + self.image_shape[:-1] + (1,))
        arrBatchGroup = np.zeros((samples.shape[0],) + (self.groups.shape[1],))
        for i, idx in enumerate(samples):
            img = PIL.Image.open(self.images[idx])
            mask = PIL.Image.open(self.masks[idx])

            if img.size != self.image_shape[:-1]:
                img = img.resize(self.image_shape[:-1])
                mask = mask.resize(self.image_shape[:-1], resample=PIL.Image.NEAREST) # pylint: disable=unexpected-keyword-arg

            img = np.array(img).astype(K.floatx())
            mask = np.array(mask).astype(K.floatx())
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if len(mask.shape) < 3:
                mask = np.expand_dims(mask, -1)
                
            if self.augmentation:
                dictAug = self.augmentation(image=np.array(img), mask=np.array(mask))
                img = dictAug['image']
                mask = dictAug['mask']
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


class SegmentationDataFrameGenerator(SegmentationDataGenerator):
    def __init__(self, dataframe, image_shape,
                 return_group=False,
                 group_encoding=None,
                 batch_size=32, 
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 min_max_scale=False,
                 augmentation=None, 
                 shuffle=True,
                 seed=None):
        """Data generator class which takes a dataframe containing paths to images and labels and group memberships. Supports augmentation with albumentation pipelines.
        Iterating over this generator returns batches of (input, groups), labels

        Args:
            dataframe (pandas.DataFrame): dataframe with columns image, mask, subject, and site
            image_dir (str): path to directory containing images
            mask_dir (str): path to directory containing binary masks
            image_shape (tup): shape of images to output
            return_group (bool): Whether to return the group membership of each image as an additional array
            group_encoding (list): List containing the order of group names for conversion to one-hot encoding
            batch_size (int, optional): Batch size. Defaults to 32.
            samplewise_center (bool, optional): Normalize each sample to zero mean. Defaults to False.
            samplewise_std_normalization (bool, optional): Normalize each sample to unit s.d. Defaults to False.
            min_max_scale (bool, optional): Scale each sample to [0, 1], cannot be used with samplewise_center and samplewise_std_normalization. Defaults to None.
            augmentation (albumentation pipeline, optional): Augmentation pipeline. Defaults to None.
            shuffle (bool, optional): Shuffle data after each epoch. Defaults to True.
            seeed (int, optional): Random seed for shuffling data.

        Raises:
            ValueError: [description]
        """        

        self.batch_size = batch_size
        self.dataframe = dataframe
        self.image_shape = image_shape
        self.return_group = return_group
        self.augmentation = augmentation
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
        if group_encoding:
            self.group_encoder = OneHotEncoder(categories=[group_encoding], drop=None, sparse=False, dtype=K.floatx())
        else:
            self.group_encoder = OneHotEncoder(drop=None, sparse=False, dtype=K.floatx())
        arrGroups = self.dataframe['group'].values.reshape((-1, 1))
        self.groups = self.group_encoder.fit_transform(arrGroups)
        
        if shuffle:
            self.on_epoch_end() # shuffle samples
        else:
            self.index = np.arange(len(self.images))