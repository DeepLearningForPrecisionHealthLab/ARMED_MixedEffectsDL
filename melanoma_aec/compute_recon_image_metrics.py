import os
import argparse
import cv2
import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
import tensorflow as tf

from medl.models import autoencoder_classifier
from medl.misc import expand_data_path, expand_results_path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to .npz file containing data.')
parser.add_argument('--metadata', type=str, required=True, help='Path to .csv file containing image metadata.')
parser.add_argument('--output_dir', type=str, required=True, help='Output path')
parser.add_argument('--model', type=str, choices=['conventional', 'adversarial', 'mixedeffects'], 
                    required=True, help='Model type')
parser.add_argument('--gpu', type=int, help='GPU to use. Defaults to all.')
args = parser.parse_args()

if args.gpu:
    set_gpu(args.gpu)

# Load data
data = np.load(expand_data_path(args.data))
dfMetadata = pd.read_csv(expand_data_path(args.metadata), index_col=0)

strOutputDir = expand_results_path(args.output_dir)

if args.model == 'conventional': 
    model = autoencoder_classifier.BaseAutoencoderClassifier()
    model.compile()
    _ = model.predict(data['images'], steps=1, batch_size=32)   
elif args.model == 'adversarial':
    model = autoencoder_classifier.DomainAdversarialAEC(n_clusters=data['cluster'].shape[1])
    model.compile()
    _ = model.predict((data['images'], data['cluster']), steps=1, batch_size=32)    
    
model.load_weights(os.path.join(strOutputDir, 'weights.h5'))

def compute_metrics(img: np.array):
    brightness = img.mean()
    contrast = img.std()
    sharpness = cv2.Laplacian(img, cv2.CV_32F).var()
    snr = brightness/contrast
    return {'Brightness': brightness,
            'Contrast': contrast,
            'Sharpness': sharpness,
            'SNR': snr}

# dfImageQuality = pd.DataFrame(columns=['Date', 'Brightness', 'Contrast', 'Sharpness', 'SNR'])
# lsMetrics = []
lsRecons = []
nImages = data['images'].shape[0]
nBatches = int(np.ceil(nImages / 1000))

for iBatch in tqdm.tqdm(range(nBatches)):
    iStart = 1000 * iBatch
    iEnd = np.min([1000 * (iBatch + 1), nImages])
    
    if args.model == 'conventional':
        data_in = data['images'][iStart:iEnd,]
    else:
        data_in = (data['images'][iStart:iEnd,], data['cluster'][iStart:iEnd,])

    arrRecons = model.predict(data_in, batch_size=32)[0]        
    # lsMetrics += [compute_metrics(arrRecons[i,]) for i in range(arrRecons.shape[0])]
    lsRecons += [arrRecons]
       
arrRecons = np.concatenate(lsRecons, axis=0)

with mp.Pool(8) as pool:
    lsMetrics = list(tqdm.tqdm(pool.imap(compute_metrics, arrRecons)))

dfMetrics = pd.DataFrame(lsMetrics)
dfMetrics.index = dfMetadata.index
dfMetrics['Date'] = dfMetadata['date'].values
dfMetrics.to_csv(os.path.join(strOutputDir, 'recon_image_metrics.csv'))                
