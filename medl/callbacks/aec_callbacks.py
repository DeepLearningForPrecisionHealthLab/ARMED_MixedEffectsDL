import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def make_save_model_callback(model, output_dir):
    
    def _save_model(epoch, logs):
        model.save_weights(os.path.join(output_dir, f'epoch{epoch+1:03d}_saved_model'))
        
    return _save_model


def make_recon_figure_callback(images, model, output_dir, clusters=None, mixedeffects=False):
    """Generate a callback function that produces a figure with example 
    reconstructions. The figure optionally includes the reconstructions 
    with and without cluster-specific effects.

    Args:
        images (np.array): images
        clusters (np.array): one-hot encoded cluster design matrix
        model (tf.keras.Model): model
        output_dir (str): output path
        mixedeffects (bool): include recons w/ and w/o random effects
    """    
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    if mixedeffects:
        
        def _recon_images(epoch, logs):
            # Callback function for saving example reconstruction images after each epoch
            fig, ax = plt.subplots(4, 9, figsize=(9, 4),
                                gridspec_kw={'hspace': 0.3, 'width_ratios': [1] * 8 + [0.2]})  
        
            arrReconME, arrReconFE, _, _ = model.predict((images, clusters))
            arrReconDiff = arrReconME - arrReconFE
            vmax = np.abs(arrReconDiff).max()

            for iImg in range(8):
                ax[0, iImg].imshow(images[iImg,], cmap='gray')
                ax[1, iImg].imshow(arrReconFE[iImg,], cmap='gray')
                ax[2, iImg].imshow(arrReconME[iImg,], cmap='gray')
                ax[3, iImg].imshow(arrReconDiff[iImg,], cmap='coolwarm', vmin=-vmax, vmax=vmax)
                
                ax[0, iImg].axis('off')
                ax[1, iImg].axis('off')
                ax[2, iImg].axis('off')
                ax[3, iImg].axis('off')
            
            ax[0, 0].text(-0.2, 0.5, 'Original', transform=ax[0, 0].transAxes, va='center', ha='center', rotation=90)    
            ax[1, 0].text(-0.2, 0.5, 'Recon: FE', transform=ax[1, 0].transAxes, va='center', ha='center', rotation=90)
            ax[2, 0].text(-0.2, 0.5, 'Recon: ME', transform=ax[2, 0].transAxes, va='center', ha='center', rotation=90)    
            ax[3, 0].text(-0.2, 0.5, '(ME - FE)', transform=ax[3, 0].transAxes, va='center', ha='center', rotation=90)
            
            for a in ax[:, -1]:
                a.remove()
            
            axCbar = fig.add_subplot(ax[0, -1].get_gridspec()[:, -1])
            fig.colorbar(ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax),
                                        cmap='coolwarm'),
                        cax=axCbar)
            axCbar.set_ylabel('Difference (ME - FE)')
            
            fig.tight_layout(w_pad=0.1, h_pad=0.1)
            fig.savefig(os.path.join(output_dir, f'epoch{epoch+1:03d}.png'))
            plt.close(fig)
    
    else:
        def _recon_images(epoch, logs):
            # Callback function for saving example reconstruction images after each epoch
            fig, ax = plt.subplots(2, 8, figsize=(8, 2))  
        
            if clusters is not None:
                arrRecon = model.predict((images, clusters))[0]
            else:
                arrRecon = model.predict(images)[0]
            
            for iImg in range(8):
                ax[0, iImg].imshow(images[iImg,], cmap='gray')
                ax[1, iImg].imshow(arrRecon[iImg,], cmap='gray')
                
                ax[0, iImg].axis('off')
                ax[1, iImg].axis('off')
            
            ax[0, 0].text(-0.2, 0.5, 'Original', transform=ax[0, 0].transAxes, va='center', ha='center', rotation=90)    
            ax[1, 0].text(-0.2, 0.5, 'Recon', transform=ax[1, 0].transAxes, va='center', ha='center', rotation=90)
                                
            fig.tight_layout(w_pad=0.1, h_pad=0.1)
            fig.savefig(os.path.join(output_dir, f'epoch{epoch+1:03d}.png'))
            plt.close(fig)
            
    return _recon_images

def make_compute_latents_callback(model, images, image_metadata, output_dir):

    def _compute_latents(epoch, logs):
        # callback function for computing latent reps for all training images and saving to a pkl file
        arrLatents = model.predict(images)
        dfLatents = pd.DataFrame(arrLatents, index=image_metadata['image'].values)
        dfLatents.to_pickle(os.path.join(output_dir, f'epoch{epoch+1:03d}_latents.pkl'))
        
        db = davies_bouldin_score(dfLatents, image_metadata['date'])
        ch = calinski_harabasz_score(dfLatents, image_metadata['date'])

        print(f'\nClustering scores:'
            f'\n\tDavies-Bouldin (higher is better): {db}'
            f'\n\tCalinski-Harabasz (lower is better): {ch}'
        )
        
        # Append to file
        with open(os.path.join(output_dir, 'clustering_scores.csv'), 'a') as f:
            if epoch == 0:
                f.write('DB,CH\n')
            f.write(f'{db},{ch}\n')
            
        
    return _compute_latents