import os
import argparse
import json
import numpy as np
import pandas as pd

from medl.misc import expand_data_path, expand_results_path

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def _shuffle_data(data_dict):
    arrIdx = np.arange(data_dict['images'].shape[0])
    np.random.seed(64)
    np.random.shuffle(arrIdx)
    return {k: v[arrIdx,] for k, v in data_dict.items()}

def train_model(model_type: str, 
                data_train: dict, 
                data_val: dict, 
                train_metadata: pd.DataFrame,
                output_dir: str,
                n_latent_dims: int=56,
                epochs: int=10,
                verbose: bool=True,
                smoketest: bool=False,
                recon_loss_weight: float=1.0,
                class_loss_weight: float=0.01,
                ):

    # Imports done inside function so that memory is allocated properly when
    # used with Ray Tune    
    import tensorflow as tf
    import tensorflow.keras.layers as tkl
    from medl.models import autoencoder_classifier
    from medl.callbacks import aec_callbacks
        
    strOutputDir = expand_results_path(output_dir)

    # Shuffle val data because Keras won't shuffle it automatically
    data_val = _shuffle_data(data_val)

    if model_type == 'conventional':        
        model = autoencoder_classifier.BaseAutoencoderClassifier()
        train_in = data_train['images']
        train_out = (data_train['images'], data_train['label'])
        val_in = data_val['images']
        val_out = (data_val['images'], data_val['label'])
          
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    #               loss={'recon': tf.keras.losses.MeanSquaredError(),
    #                     'class': tf.keras.losses.BinaryCrossentropy()},
    #               loss_weights={'recon': recon_loss_weight, 
    #                             'class': class_loss_weight},
    #               metrics={'recon': [tf.keras.metrics.MeanSquaredError(name='mse')],
    #                        'class': [tf.keras.metrics.BinaryAccuracy(name='bce')]})
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                      loss=[tf.keras.losses.MeanSquaredError(name='mse'),
                            tf.keras.losses.BinaryCrossentropy(name='bce')],
                      loss_weights=[recon_loss_weight, class_loss_weight],
                      metrics=[[],
                               [tf.keras.metrics.AUC(name='auroc')]])  
        
    elif model_type == 'adversarial':
        model = autoencoder_classifier.DomainAdversarialAEC(n_clusters=data_train['cluster'].shape[1])
        train_in = (data_train['images'], data_train['cluster'])
        train_out = (data_train['images'], data_train['label'])
        val_in = (data_val['images'], data_val['cluster'])
        val_out = (data_val['images'], data_val['label'])
        
        model.compile(loss_recon=tf.keras.losses.MeanSquaredError(),
                      loss_class=tf.keras.losses.BinaryCrossentropy(),
                      loss_adv=tf.keras.losses.BinaryCrossentropy(),
                      metric_class=tf.keras.metrics.AUC(name='auroc'),
                      metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                      opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                      opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                      loss_recon_weight=1.0,
                      loss_class_weight=0.01,
                      loss_gen_weight=0.02)
    
    elif model_type == 'mixedeffects':
        model = autoencoder_classifier.MixedEffectsAEC(n_clusters=data_train['cluster'].shape[1])
        train_in = (data_train['images'], data_train['cluster'])
        train_out = (data_train['images'], data_train['label'])
        val_in = (data_val['images'], data_val['cluster'])
        val_out = (data_val['images'], data_val['label'])
        
        model.compile(loss_recon=tf.keras.losses.MeanSquaredError(),
                      loss_class=tf.keras.losses.BinaryCrossentropy(),
                      loss_adv=tf.keras.losses.BinaryCrossentropy(),
                      metric_class=tf.keras.metrics.AUC(name='auroc'),
                      metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                      opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                      opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                      loss_recon_me_weight=1.0,
                      loss_recon_fe_weight=1.0,
                      loss_class_weight=0.01,
                      loss_gen_weight=0.02)
    
    # Separate model which only returns the encoder output
    encoder_in = tkl.Input((256, 256, 1), name='encoder_in')
    encoder_out = model.encoder(encoder_in)[-1]
    encoder = tf.keras.models.Model(encoder_in, encoder_out, name='standalone_encoder')
    
    # Get a small set of samples to generate recon figure each epoch
    arrReconSampleIdx = np.arange(data_val['images'].shape[0])
    np.random.seed(64)
    arrReconSampleIdx = np.random.choice(arrReconSampleIdx, size=8)
    arrBatchX = data_val['images'][arrReconSampleIdx,]
    arrBatchZ = data_val['cluster'][arrReconSampleIdx,]
    
    recon_images = aec_callbacks.make_recon_figure_callback(arrBatchX, model, output_dir,
                                                            clusters=None if model_type == 'conventional' else arrBatchZ,
                                                            mixedeffects=model_type == 'mixedeffects')
    
    compute_latents = aec_callbacks.make_compute_latents_callback(encoder, data_train['images'],
                                                                  train_metadata, output_dir)
    
    lsCallbacks = [tf.keras.callbacks.CSVLogger(os.path.join(strOutputDir, 'training_log.csv')),
                   tf.keras.callbacks.LambdaCallback(on_epoch_end=recon_images),
                   tf.keras.callbacks.LambdaCallback(on_epoch_end=compute_latents),
                   tf.keras.callbacks.ModelCheckpoint(os.path.join(strOutputDir, 'epoch{epoch:03d}_weights.h5'),
                                                      save_weights_only=True)]
    
    history = model.fit(train_in, train_out,
                        epochs=epochs,
                        verbose=verbose,
                        batch_size=32,
                        validation_data=(val_in, val_out),
                        shuffle=True,
                        steps_per_epoch=10 if smoketest else None,
                        validation_steps=10 if smoketest else None,
                        callbacks=lsCallbacks)
                
    # model.save_weights(os.path.join(output_dir, 'weights.h5'))
    
    dfHistory = pd.DataFrame(history.history)
    dictResults = dfHistory.iloc[-1].to_dict()
    
    # Compute clustering metrics
    arrLatents = encoder.predict(data_train['images'])
    db = davies_bouldin_score(arrLatents, train_metadata['date'])
    ch = calinski_harabasz_score(arrLatents, train_metadata['date'])
    
    dictResults.update(db=db, ch=ch)
    
    return dictResults


def test_model(model_type: str, 
               data: dict, 
               output_dir: str,
               smoketest: bool=False,
               recon_loss_weight: float=1.0,
               class_loss_weight: float=0.01,
              ):
    # Imports done inside function so that memory is allocated properly when
    # used with Ray Tune    
    import tensorflow as tf
    import tensorflow.keras.layers as tkl
    from medl.models import autoencoder_classifier
    from medl.callbacks import aec_callbacks
        
    strOutputDir = expand_results_path(output_dir)
    data = _shuffle_data(data)

    if model_type == 'conventional':
        model = autoencoder_classifier.BaseAutoencoderClassifier()
        data_in = data['images']
        data_out = (data['images'], data['label'])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                    loss=[tf.keras.losses.MeanSquaredError(name='mse'),
                            tf.keras.losses.BinaryCrossentropy(name='bce')],
                    loss_weights=[recon_loss_weight, class_loss_weight],
                    metrics=[[],
                            [tf.keras.metrics.AUC(name='auroc')]]) 
    
    elif model_type == 'adversarial':
        model = autoencoder_classifier.DomainAdversarialAEC(n_clusters=data['cluster'].shape[1])
        data_in = (data['images'], data['cluster'])
        data_out = (data['images'], data['label'])
        
        model.compile(loss_recon=tf.keras.losses.MeanSquaredError(),
                      loss_class=tf.keras.losses.BinaryCrossentropy(),
                      loss_adv=tf.keras.losses.BinaryCrossentropy(),
                      metric_class=tf.keras.metrics.AUC(name='auroc'),
                      metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                      opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                      opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                      loss_recon_weight=1.0,
                      loss_class_weight=0.01,
                      loss_gen_weight=0.02)
    
    elif model_type == 'mixedeffects':
        model = autoencoder_classifier.MixedEffectsAEC(n_clusters=data['cluster'].shape[1])
        data_in = (data['images'], data['cluster'])
        data_out = (data['images'], data['label'])
        
        model.compile(loss_recon=tf.keras.losses.MeanSquaredError(),
                      loss_class=tf.keras.losses.BinaryCrossentropy(),
                      loss_adv=tf.keras.losses.BinaryCrossentropy(),
                      metric_class=tf.keras.metrics.AUC(name='auroc'),
                      metric_adv=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                      opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                      opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                      loss_recon_me_weight=1.0,
                      loss_recon_fe_weight=1.0,
                      loss_class_weight=0.01,
                      loss_gen_weight=0.02)
    
    _ = model.predict(data_in, steps=1, batch_size=32)    
    model.load_weights(os.path.join(output_dir, 'weights.h5'))
            
    dictMetrics = model.evaluate(data_in, data_out, 
                                 batch_size=32,
                                 steps=10 if smoketest else None,
                                 return_dict=True)
                
    return dictMetrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_metadata', type=str, 
                        default='melanoma/4pdx6dates/data_train.csv',
                        help='Path to CSV table containing training image metadata.')
    parser.add_argument('--training_data', type=str, 
                        default='melanoma/4pdx6dates/data_train.npz',
                        help='Path to .npz file containing training images.')
    parser.add_argument('--val_data', type=str, 
                        default='melanoma/4pdx6dates/data_val.npz',
                        help='Path to .npz file containing validation images.')
    parser.add_argument('--test_data', type=str, 
                        default='melanoma/4pdx6dates/data_test.npz',
                        help='Path to .npz file containing test images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--model_type', type=str, choices=['conventional', 'adversarial', 'mixedeffects'],
                        required=True, help='Model type.')
    parser.add_argument('--epochs', type=int, default=10, help='Training duration. Defaults to 10')
    parser.add_argument('--do_test', action='store_true', help='Evaluate on test')
    parser.add_argument('--skip_train', action='store_true', help='Skip training and load saved model.')
    parser.add_argument('--loss_recon_weight', type=float, default=1.0, help='Reconstruction loss weight.')
    parser.add_argument('--loss_class_weight', type=float, default=0.01, help='Classification loss weight.')

    parser.add_argument('--verbose', type=int, default=1, help='Show training progress.')
    parser.add_argument('--gpu', type=int, help='GPU to use. Defaults to all.')
    parser.add_argument('--smoketest', action='store_true', 
                        help='For testing purposes, train just 10 batches per epoch.')

    args = parser.parse_args()

    if args.gpu:
        from medl.tfutils import set_gpu
        set_gpu(args.gpu)
        
    strOutputDir = expand_results_path(args.output_dir, make=True)
    
    if not args.skip_train:    
        strTrainDataPath = expand_data_path(args.training_data)
        strTrainMetaDataPath = expand_data_path(args.training_metadata)
        strValDataPath = expand_data_path(args.val_data)
        dfTrainMetadata = pd.read_csv(strTrainMetaDataPath, index_col=0)

        dictDataTrain = np.load(strTrainDataPath)
        dictDataVal = np.load(strValDataPath)
    
        dictMetrics = train_model(model_type=args.model_type, 
                                data_train=dictDataTrain, 
                                data_val=dictDataVal,
                                train_metadata=dfTrainMetadata, 
                                output_dir=strOutputDir,
                                epochs=args.epochs,
                                verbose=args.verbose == 1,
                                smoketest=args.smoketest,
                                recon_loss_weight=args.loss_recon_weight,
                                class_loss_weight=args.loss_class_weight)
        
        print(dictMetrics)
        
    if args.do_test:
        strTestDataPath = expand_data_path(args.test_data)
        dictDataTest = np.load(strTestDataPath)

        dictMetrics = test_model(model_type=args.model_type,
                                 data=dictDataTest,
                                 output_dir=strOutputDir,
                                 smoketest=args.smoketest,
                                 recon_loss_weight=args.loss_recon_weight,
                                 class_loss_weight=args.loss_class_weight)
        
        with open(os.path.join(strOutputDir, 'test_metrics.json'), 'w') as f:
            json.dump(dictMetrics, f, indent=4)