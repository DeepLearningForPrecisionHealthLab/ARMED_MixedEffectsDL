'''
Train mixed effects adversarial autoencoder on melanoma cell images. 
-----
Kevin Nguyen
1/18/2021
'''

TRAIN = '/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma/filtered/data_train.csv'
VAL = '/archive/bioinformatics/DLLab/KevinNguyen/data/melanoma/filtered/data_val.csv'
EPOCHS = 200 # training epochs
LATENT = 56 # number of latent dimensions
VALFREQ = 1 # number of epochs to train before evaluating on val data
VALITER = 20 # number of validation batches to use for evaluation
SAVEFREQ = 20 # how often to save model weights
GPU = -1 # GPU to use, set to -1 to use all

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
sys.path.append('../')
from medl.datagenerator import AutoencoderGroupedDataGenerator # pylint: disable=import-error
from me_aae import me_adversarial_autoencoder

args = argparse.ArgumentParser()
args.add_argument('--intercept', '-i', type=str, help='y to Include random intercept')
args.add_argument('--slope', '-s', type=str, help='y to Include random slope')
args.add_argument('--outdir', '-o', type=str, required=True, help='output path')
arguments = args.parse_args()

INT = arguments.slope == 'y'
SLOPE = arguments.slope == 'y'
OUTPUTDIR = arguments.outdir

if GPU > -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

os.makedirs(OUTPUTDIR, exist_ok=True)
# Record the data file paths used
with open(os.path.join(OUTPUTDIR, 'paths.txt'), 'w') as f:
    f.write(f'train,{TRAIN}\nval,{VAL}')

dfTrain = pd.read_csv(TRAIN, index_col=0)
dfVal = pd.read_csv(VAL, index_col=0)

arrGroupOrder = dfTrain['date'].unique()
arrGroupOrder.sort()
train_data = AutoencoderGroupedDataGenerator(dfTrain['image'].values, dfTrain['date'].values, (256, 256, 1), 
                                             group_encoding=arrGroupOrder,
                                             return_group=True,
                                             max_value=255.0)
val_data = AutoencoderGroupedDataGenerator(dfVal['image'].values, dfVal['date'].values, (256, 256, 1), 
                                            group_encoding=arrGroupOrder,
                                            return_group=True,
                                            max_value=255.0)

autoencoder, adversary, encoder = me_adversarial_autoencoder(n_groups=arrGroupOrder.size,
                                                            n_latent_dims=LATENT, 
                                                            recon_loss_weight=0.9, 
                                                            adv_loss_weight=0.1,
                                                            random_slope=SLOPE,
                                                            random_int=INT)

# Set up text log and tensorboard files
strLogPath = os.path.join(OUTPUTDIR, 'log.txt')
with open(strLogPath, 'w') as f:
    f.write('adv_loss, adv_acc_real, adv_acc_fake, ae_loss, ae_recon_mse, ae_gen_bce, val_adv_acc_real, val_adv_acc_fake, val_ae_recon_mse, val_ae_gen_acc')

strTrainLogDir = os.path.join(OUTPUTDIR, 'tensorboard', 'train')
strValLogDir = os.path.join(OUTPUTDIR, 'tensorboard', 'val')
tbTrain = tf.summary.create_file_writer(strTrainLogDir)
tbVal = tf.summary.create_file_writer(strValLogDir)

for iEpoch in range(EPOCHS):
    print('----- EPOCH', iEpoch, '-----\n', flush=True)
    # Train on each batch
    for iBatch, ((arrBatch, arrGroups), _) in tqdm(enumerate(train_data), total=len(train_data)):
        nBatchSize = arrBatch.shape[0]
        arrYReal = np.ones((nBatchSize,))
        arrYFake = np.zeros((nBatchSize,))
        arrLatentReal = np.random.randn(nBatchSize, LATENT)
        # Train adversary
        arrLatentFake = encoder.predict(arrBatch)
        fAdvLossTrainReal, fAdvAccTrainReal = adversary.train_on_batch(arrLatentReal, arrYReal)
        fAdvLossTrainFake, fAdvAccTrainFake = adversary.train_on_batch(arrLatentFake, arrYFake)
        fAdvLossTrain = 0.5 * (fAdvLossTrainReal + fAdvLossTrainFake)
        # fAdvAccTrain = 0.5 * (fAdvAccTrainReal + fAdvAccTrainFake)
        # Train autoencoder 
        fAeLossTrain, _, _, fAeMSETrain, fAeAccTrain = autoencoder.train_on_batch([arrBatch, arrGroups], [arrBatch, arrYReal])

    print('\nadv_loss: {:.05f}, adv_acc_real: {:.05f}, adv_acc_fake: {:.05f}, ae_loss: {:.05f}, ae_recon_mse: {:.05f}, ae_gen_acc: {:.05f}'.format(fAdvLossTrain,
                                                                                                                    fAdvAccTrainReal,
                                                                                                                    fAdvAccTrainFake,
                                                                                                                    fAeLossTrain,
                                                                                                                    fAeMSETrain,
                                                                                                                    fAeAccTrain))                                    
    # Update tensorboard
    with tbTrain.as_default(): # pylint: disable=not-context-manager
        tf.summary.scalar('adversary_loss', fAdvLossTrain, step=iEpoch)    
        tf.summary.scalar('adversary_acc/real', fAdvAccTrainReal, step=iEpoch)
        tf.summary.scalar('adversary_acc/fake', fAdvAccTrainFake, step=iEpoch)
        tf.summary.scalar('autoencoder_loss', fAeLossTrain, step=iEpoch)
        tf.summary.scalar('autoencoder_MSE', fAeMSETrain, step=iEpoch)
        tf.summary.scalar('autoencoder_acc', fAeAccTrain, step=iEpoch)
    train_data.on_epoch_end() # Shuffle batches

    if iEpoch % VALFREQ == 0:
        print('---Evaluating---', flush=True)
        fAdvAccValReal = 0
        fAdvAccValFake = 0
        fAeMSEVal = 0
        fAeAccVal = 0
        for iBatchVal, ((arrBatchVal, arrGroupsVal), _) in enumerate(val_data):
            if iBatchVal >= VALITER:
                continue
            nBatchSize = arrBatchVal.shape[0]
            arrYReal = np.ones((nBatchSize,))
            arrYFake = np.zeros((nBatchSize,))
            arrLatentReal = np.random.randn(nBatchSize, LATENT)
            arrLatentFake = encoder.predict(arrBatchVal)

            # Evaluate adversary
            _, accR = adversary.evaluate(arrLatentReal, arrYReal, verbose=0)
            _, accF = adversary.evaluate(arrLatentFake, arrYFake, verbose=0)
            # fAdvAccVal += 0.5 * (fAdvAccValReal + fAdvAccValFake)
            fAdvAccValReal += accR
            fAdvAccValFake += accF

            # Evaluate autoencoder
            _, _, _, mse, acc = autoencoder.evaluate([arrBatchVal, arrGroupsVal], [arrBatchVal, arrYReal], verbose=0)
            fAeMSEVal += mse
            fAeAccVal += acc
        fAdvAccValReal /= len(val_data)
        fAdvAccValFake /= len(val_data)
        fAeMSEVal /= len(val_data)
        fAeAccVal /= len(val_data)

        print('val_adv_acc_real: {:.05f}, val_adv_acc_fake: {:.05f}, val_ae_recon_mse:  {:.05f}, val_ae_gen_acc: {:.05f}'.format(fAdvAccValReal, 
                                                                                                                            fAdvAccValFake,
                                                                                                                            fAeMSEVal, 
                                                                                                                            fAeAccVal))

        # Save example reconstruction images
        fig, ax = plt.subplots(2, 8)
        arrBatchVal, arrGroupsVal = val_data[0][0]
        arrReconVal, _ = autoencoder.predict([arrBatchVal, arrGroupsVal])
        for iImg in range(8):
            ax[0, iImg].imshow(arrBatchVal[iImg,], cmap='gray')
            ax[1, iImg].imshow(arrReconVal[iImg,], cmap='gray')
            ax[0, iImg].axis('off')
            ax[1, iImg].axis('off')
        fig.tight_layout(w_pad=0.2, h_pad=0.2)
        fig.savefig(os.path.join(OUTPUTDIR, f'epoch{iEpoch:03d}.png'))
        plt.close(fig)
        
        # Update tensorboard
        with tbVal.as_default(): # pylint: disable=not-context-manager   
            tf.summary.scalar('adversary_acc/real', fAdvAccValReal, step=iEpoch)
            tf.summary.scalar('adversary_acc/fake', fAdvAccValFake, step=iEpoch)
            tf.summary.scalar('autoencoder_MSE', fAeMSEVal, step=iEpoch)
            tf.summary.scalar('autoencoder_acc', fAeAccVal, step=iEpoch)
    else:
        fAdvAccValReal = ''
        fAdvAccValFake = ''
        fAeMSEVal = ''
        fAeAccVal = ''

    # Save model checkpoint 
    if (iEpoch % SAVEFREQ == 0) | (iEpoch == (EPOCHS-1)):
        fBestMSE = fAeMSEVal
        print('Saving model checkpoint', flush=True)
        autoencoder.save_weights(os.path.join(OUTPUTDIR, f'epoch{iEpoch}_autoencoder.h5'))
        adversary.save_weights(os.path.join(OUTPUTDIR, f'epoch{iEpoch}_adversary.h5'))
        encoder.save_weights(os.path.join(OUTPUTDIR, f'epoch{iEpoch}_encoder.h5'))

    lsMetrics = [fAdvLossTrain, fAdvAccTrainReal, fAdvAccTrainFake, fAeLossTrain, fAeMSETrain, fAeAccTrain, fAdvAccValReal, fAdvAccValFake, fAeMSEVal, fAeAccVal]
    with open(strLogPath, 'a') as f:
        f.write('\n')
        f.write(','.join([str(x) for x in lsMetrics]))
        
    print('\n')