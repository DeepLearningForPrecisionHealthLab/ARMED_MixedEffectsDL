'''
Custom Keras-compatible data generators for data with associated group labels
'''
import os
import re
from nibabel.nifti1 import Nifti1Image
import numpy as np
import glob
import PIL
import nibabel as nib
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K

class GroupedDataGenerator(Sequence):
    def __init__(self, arrX, arrY, arrGroups, 
                 batch_size=32, 
                 label_type=None,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 min_max_scale=False,
                 augmentation=None, 
                 shuffle=True,
                 seed=None):
        """Data generator class for data with associated group labels. Supports augmentation with albumentation pipelines.
        Iterating over this generator returns batches of (input, groups), labels

        Args:
            arrX (np.array): n_samples x ... array of inputs
            arrY (np.array): n_samples x ... array of labels
            arrGroups (np.array): n_samples x ... array of group memberships of each sample (this gets converted into a one-hot design matrix)
            batch_size (int, optional): Batch size. Defaults to 32.
            label_type (str, optional): Set to 'mask' if performing segmentation with binary masks. Defaults to None.
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
        self.label_type = label_type
        self.arrX = arrX.astype(K.floatx())
        self.arrY = arrY.astype(K.floatx())
        self.group_encoder = OneHotEncoder(drop=None, sparse=False, dtype=K.floatx(), handle_unknown='ignore')
        arrOnehot = self.group_encoder.fit_transform(arrGroups)
        self.arrGroupsCat = arrGroups
        self.arrGroups = arrOnehot
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.n_total_epochs_seen = -1
        self.seed = seed

        if min_max_scale & (samplewise_center | samplewise_std_normalization):
            raise ValueError('min_max_scale cannot be used if samplewise_center or samplewise_std_normalization are True')
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.min_max_scale = min_max_scale

        self.on_epoch_end() # shuffle samples

    def __len__(self):
        return int(np.ceil(self.arrX.shape[0] / self.batch_size))

    def __getitem__(self, batch_index):
        batch_end = np.min([((batch_index + 1) * self.batch_size), self.index.shape[0]])
        samples = self.index[(batch_index * self.batch_size):batch_end]
        return self.__get_data(samples)

    def __get_data(self, samples):
        arrBatchX = np.zeros((samples.shape[0],) + self.arrX.shape[1:])
        arrBatchY = np.zeros((samples.shape[0],) + self.arrY.shape[1:])
        arrBatchGroups = np.zeros((samples.shape[0],) + self.arrGroups.shape[1:])
        for i, idx in enumerate(samples):
            arrSampleX = self.arrX[idx,]
            arrSampleY = self.arrY[idx,]
            if self.augmentation:
                if self.label_type == 'mask':
                    dictAug = self.augmentation(image=arrSampleX, mask=arrSampleY)
                    arrSampleX = dictAug['image']
                    arrSampleY = dictAug['mask']
                else:
                    arrSampleX = self.augmentation(image=arrSampleX)['image']
            if self.samplewise_center:
                arrSampleX -= arrSampleX.mean()
            if self.samplewise_std_normalization:
                arrSampleX /= arrSampleX.std()
            if self.min_max_scale:
                arrSampleX -= arrSampleX.min()
                arrSampleX /= arrSampleX.max()
            arrBatchX[i,] = arrSampleX
            arrBatchY[i,] = arrSampleY
            arrBatchGroups[i,] = self.arrGroups[idx,]
        return (arrBatchX, arrBatchGroups), arrBatchY

    def on_epoch_end(self):
        self.index = np.arange(self.arrX.shape[0])
        if self.seed:
            np.random.seed(self.seed + self.n_total_epochs_seen)
        if self.shuffle:
            np.random.shuffle(self.index)
        self.n_total_epochs_seen += 1
        return super().on_epoch_end()

    def set_dummy_encoder(self, tupShape):
        ''' Instead of creating the one-hot encoded design matrix based on the 
        group membership of each sample, use a dummy design matrix of all zeros.
        This should be used for test data, when none of the test data comes from 
        the same groups seen in the training data and therefore the learned random 
        effects cannot be applied. Pass in a tuple containing the number of groups 
        present in the training data.

        Args:
            tupShape (tuple): (n_groups,)
        '''
        self.group_encoder = lambda x: np.zeros((x.shape[0],) + tupShape)
        self.arrGroups = self.group_encoder(self.arrGroupsCat)


class AutoencoderGroupedDataGenerator(Sequence):
    def __init__(self, image_list, groups, image_shape,
                 return_group=False,
                 group_encoding=None,
                 batch_size=32, 
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 min_max_scale=False,
                 max_value=None,
                 augmentation=None, 
                 shuffle=True,
                 seed=None):
        """Data generator class for autoencoder image data with associated group labels. Supports augmentation with albumentation pipelines.
        Iterating over this generator returns batches of image, image or (image, groups), image if return_group == True

        Args:
            image_list (str): list of image paths
            image_shape (tup): shape to conform all images
            return_group (bool): whether to return the group design matrix with each batch
            group_encoding (list, array): List containing the order of group names for conversion to one-hot encoding
            batch_size (int, optional): Batch size. Defaults to 32.
            samplewise_center (bool, optional): Normalize each sample to zero mean. Defaults to False.
            samplewise_std_normalization (bool, optional): Normalize each sample to unit s.d. Defaults to False.
            min_max_scale (bool, optional): Scale each sample to [0, 1], cannot be used with samplewise_center and samplewise_std_normalization. Defaults to None.
            augmentation (albumentation pipeline, optional): Augmentation pipeline. Defaults to None.
            shuffle (bool, optional): Shuffle data after each epoch. Defaults to True.
            seed (int, optional): Random seed for shuffling data.

        Raises:
            ValueError: min_max_scale and samplewise normalization cannot be used simultaneously
        """  
        self.batch_size = batch_size
        self.images = image_list
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
        self.max_value = max_value

        # Determine image + mask pairings
        if group_encoding is not None:
            group_encoding = [group_encoding]
        else:
            group_encoding = 'auto'
        self.group_encoder = OneHotEncoder(categories=group_encoding, drop=None, sparse=False, dtype=K.floatx())
        if isinstance(groups, np.ndarray):
            self.arrGroupsCat = groups.reshape((-1, 1))
        else:
            self.arrGroupsCat = np.array(groups).reshape((-1, 1))
        self.arrGroups = self.group_encoder.fit_transform(self.arrGroupsCat)

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
        arrBatchGroup = np.zeros((samples.shape[0],) + (self.arrGroups.shape[1],))
        for i, idx in enumerate(samples):
            img = PIL.Image.open(self.images[idx])

            if img.size != self.image_shape[:-1]:
                img = img.resize(self.image_shape[:-1])

            img = np.array(img).astype(K.floatx())
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
                
            if self.augmentation:
                dictAug = self.augmentation(image=np.array(img))
                img = dictAug['image']
            if self.samplewise_center:
                img -= img.mean()
            if self.samplewise_std_normalization:
                img /= img.std()
            # if self.min_max_scale:
            #     img -= img.min()
            #     img /= img.max()
            if self.max_value:
                img /= self.max_value

            arrBatchX[i,] = img
            arrBatchGroup[i,] = self.arrGroups[idx,]
        if self.return_group:
            return (arrBatchX, arrBatchGroup), arrBatchX
        else:
            return arrBatchX, arrBatchX

    def on_epoch_end(self):
        self.index = np.arange(len(self.images))
        if self.seed:
            np.random.seed(self.seed + self.n_total_epochs_seen)
        if self.shuffle:
            np.random.shuffle(self.index)
        self.n_total_epochs_seen += 1
        return super().on_epoch_end()

    def set_dummy_encoder(self, tupShape):
        ''' Instead of creating the one-hot encoded design matrix based on the 
        group membership of each sample, use a dummy design matrix of all zeros.
        This should be used for test data, when none of the test data comes from 
        the same groups seen in the training data and therefore the learned random 
        effects cannot be applied. Pass in a tuple containing the number of groups 
        present in the training data.

        Args:
            tupShape (tuple): (n_groups,)
        '''
        self.group_encoder = lambda x: np.zeros((x.shape[0],) + tupShape)
        self.arrGroups = self.group_encoder(self.arrGroupsCat)


class SegmentationDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, image_shape,
                 return_contrast=False,
                 contrast_encoding=['t1', 't1ce', 't2', 'flair'],
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
            return_contrast (bool): Whether to return the MRI contrast of each image as an additional array
            contrast_encoding (list): List containing the order of contrast names for conversion to one-hot encoding
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
        self.return_contrast = return_contrast
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
            regmatch = re.search(r'(\d+)_(.*)_slice(\d+).png', strImgFile)
            strSubject = regmatch[1]
            strContrast = regmatch[2]
            strSlice = regmatch[3]
            strMaskPath = os.path.join(self.mask_dir, f'{strSubject}_seg_slice{strSlice}.png')
            if not os.path.exists(strMaskPath):
                raise FileNotFoundError(strMaskPath)
            return strMaskPath, strContrast
        lsMasksContrasts = [_match_mask(s) for s in self.images]
        self.masks = [t[0] for t in lsMasksContrasts]
        self.contrast_encoder = OneHotEncoder(categories=[contrast_encoding], drop=None, sparse=False, dtype=K.floatx())
        arrContrasts = np.array([t[1] for t in lsMasksContrasts], dtype='object').reshape((-1, 1))
        self.contrasts = self.contrast_encoder.fit_transform(arrContrasts)

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
        arrBatchContrast = np.zeros((samples.shape[0],) + (self.contrasts.shape[1],))
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
            arrBatchContrast[i,] = self.contrasts[idx]
        if self.return_contrast:
            return (arrBatchX, arrBatchContrast), arrBatchY
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

class SegmentationDataFrameGenerator2D(Sequence):
    def __init__(self, dataframe, image_shape, 
                 mask_shape=None, 
                 return_group=False,
                 group_encoding=None,
                 batch_size=32, 
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 min_max_scale=False,
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
            self.mask_shape = image_shape[:-1] + (1,)
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
            if len(img.shape) < 3:
                img = np.expand_dims(img, axis=-1)
            if len(mask.shape) < 3:
                mask = np.expand_dims(mask, axis=-1)

            if img.shape[:2] != self.image_shape[:2]:
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
    
