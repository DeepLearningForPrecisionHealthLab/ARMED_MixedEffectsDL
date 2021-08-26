CLUSTERS = 10
PROCS = 8

import os
import shutil

import numpy as np
import pandas as pd
import multiprocessing as mp

import tqdm
from medl.settings import DATADIR

from morphomnist import io as mio
from morphomnist import morpho, perturb

def transform_images(images: np.array, 
                     target_thickness: float=2.7, 
                     swelling_strength: float=0, 
                     swelling_radius: float=7,
                     num_fractures: int=0,
                     fracture_thickness: float=1.5,
                     n_proc=1) -> np.array:
    
    global _transform_single
    def _transform_single(idx):
        image = images[idx,]
        morphologyOrig = morpho.ImageMorphology(image, scale=4)
        fMeanThickness = morphologyOrig.mean_thickness
        
        try:
        
            # 1. Change thiccness
            # Compute % thickness change needed relative to original thickness
            fChangeRatio = (target_thickness - fMeanThickness) / fMeanThickness
            if fChangeRatio < 0:
                thicken = perturb.Thinning(amount=np.abs(fChangeRatio))
            else:
                thicken = perturb.Thickening(amount=fChangeRatio)
            arrImageThickened = morphologyOrig.downscale(thicken(morphologyOrig))
            
            # 2. Add swelling
            if swelling_strength > 0:
                morphologyThickened = morpho.ImageMorphology(arrImageThickened, scale=4)
                swell = perturb.Swelling(strength=swelling_strength, radius=swelling_radius)
                arrImageSwollen = morphologyThickened.downscale(swell(morphologyThickened))
            else:
                arrImageSwollen = arrImageThickened
            
            # 3. Add fractures
            if num_fractures > 0:
                morphologySwollen = morpho.ImageMorphology(arrImageSwollen, scale=4)
                fracture = perturb.Fracture(thickness=fracture_thickness, num_frac=num_fractures)
                arrImageFractured = morphologySwollen.downscale(fracture(morphologySwollen))
            else:
                arrImageFractured = arrImageSwollen
                
        except ValueError:
            # Morpho-MNIST sometimes throws an error when applying the perturbations, not sure why
            arrImageFractured = image
            
        return arrImageFractured
    
    nImages = images.shape[0]
    if n_proc > 1:
        with mp.Pool(n_proc) as pool:
            lsTransformed = list(tqdm.tqdm(pool.imap(_transform_single, range(nImages)), total=nImages))
            
    else:
        lsTransformed = list(tqdm.tqdm([_transform_single(x) for x in range(nImages)], total=nImages))
        
    return np.stack(lsTransformed, axis=0)

if __name__ == '__main__':

    strOrigMnistPath = os.path.join(DATADIR, 'morpho-mnist', 'plain')
    strOutputPath = os.path.join(DATADIR, 'morpho-mnist', '{}_clusters'.format(CLUSTERS))
    
    os.makedirs(strOutputPath, exist_ok=True)
    
    arrImagesOrig = mio.load_idx(os.path.join(strOrigMnistPath, 'train-images-idx3-ubyte.gz'))
    nImagesPerCluster = arrImagesOrig.shape[0] // CLUSTERS
    
    arrImagesTransformed = np.zeros_like(arrImagesOrig)
    
    arrClusterThickness = np.random.normal(loc=2.7, scale=1, size=(CLUSTERS,))
    arrClusterSwelling = np.random.normal(loc=3, scale=1.0, size=(CLUSTERS,))
    arrClusterSwelling = np.clip(arrClusterSwelling, 0, 5)
    arrClusterFractures = np.random.normal(loc=3, scale=1.0, size=(CLUSTERS,))
    arrClusterFractures = np.clip(arrClusterFractures, 1, 10).astype(int)
    
    df = pd.DataFrame({'thickness': arrClusterThickness,
                       'swelling_strength': arrClusterSwelling,
                       'fractures': arrClusterFractures})
    
    print(df)
    df.to_csv(os.path.join(strOutputPath, 'transformations.csv'))
            
    for iCluster in range(CLUSTERS):
        iStart = iCluster * nImagesPerCluster
        iEnd = (iCluster + 1) * nImagesPerCluster
        
        arrImagesCluster = transform_images(arrImagesOrig[iStart:iEnd,],
                                            target_thickness=arrClusterThickness[iCluster],
                                            swelling_strength=arrClusterSwelling[iCluster],
                                            num_fractures=arrClusterFractures[iCluster],
                                            n_proc=PROCS)
        
        arrImagesTransformed[(iCluster * nImagesPerCluster):((iCluster + 1) * nImagesPerCluster)] = arrImagesCluster
        
    mio.save_idx(arrImagesTransformed, os.path.join(strOutputPath, 'training_images.gz'))
    # Copy original labels and test data as is
    shutil.copy(os.path.join(strOrigMnistPath, 'train-labels-idx1-ubyte.gz'), 
                os.path.join(strOutputPath, 'training_labels.gz'))
    shutil.copy(os.path.join(strOrigMnistPath, 't10k-images-idx3-ubyte.gz'), 
                os.path.join(strOutputPath, 'test_images.gz'))
    shutil.copy(os.path.join(strOrigMnistPath, 't10k-labels-idx1-ubyte.gz'), 
                os.path.join(strOutputPath, 'test_labels.gz'))