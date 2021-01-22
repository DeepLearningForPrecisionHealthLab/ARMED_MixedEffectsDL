'''
Use trained encoder to generate latent representations on test data.
'''

DATA = '/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma/filtered/data_{}.csv'
WEIGHTS = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/melanoma_me_autoencoder_filtered_20210121/random_int/epoch140_{}.h5'
SAVEPATH = '/archive/bioinformatics/DLLab/KevinNguyen/results/MEDL/melanoma_me_autoencoder_filtered_20210121/random_int/latent_reps_epoch140_{}.npy'
LATENT = 56

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append('../')
from medl.models import aae # pylint: disable=import-error
from medl.datagenerator import AutoencoderGroupedDataGenerator # pylint: disable=import-error

autoencoder, adversary, encoder = aae.adversarial_autoencoder(n_latent_dims=LATENT)
del adversary
del autoencoder
encoder.load_weights(WEIGHTS.format('encoder'))

for strPartition in ['train', 'val', 'test']:
    dfData = pd.read_csv(DATA.format(strPartition), index_col=0)
    datagen = AutoencoderGroupedDataGenerator(dfData['image'].values, dfData['date'].values, (256, 256, 1), 
                                            min_max_scale=True)

    lsLatent = []
    for arrX, _ in datagen:
        arrLabels = np.ones((arrX.shape[0], 1))
        lsLatent += [encoder.predict(arrX)]
    arrLatent = np.concatenate(lsLatent, axis=0)

    np.save(SAVEPATH.format(strPartition), arrLatent)