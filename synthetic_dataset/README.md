# Synthetic datasets

## Setup 
Ensure that the root of this repository is on your `PYTHONPATH` environmental variable. For the digit classifciation example, we also need [Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST) downloaded and on the `PYTHONPATH`. 

See `.env` for an example `PYTHONPATH`. 

## Spiral classification
Classic two-dimensional nonlinear classification benchmark. Data points are sampled along two (or more) spiral functions and the neural network must learn to classify points by their spiral. Random effects are simulated by dividing the data into clusters and random varying the spiral radius in each cluster. 

### Main script
The `spiral_classification_main.py` runs a simple comparison of 4 model types, with leave-one-cluster-out cross-validation. To get started:

```python
python spiral_classification_main.py --output_dir </path/to/output/location>
``` 

See `python spiral_classification_main.py --help` for details on different simulation parameters. 

### Associated scripts
`spiral_classification_paramgrid.py` performs a sensitivity analysis, basically running `spiral_classification_main.py` over a grid of simulation parameters. 

## Morpho-MNIST digit classification
[Castro et al.](https://jmlr.org/papers/v20/19-033.html) published a method (Morpho-MNIST) for simulating morphological variations in the classic MNIST digit classification problem. These morphological variations include stroke thickness, local swelling (dilation of part of the stroke), and fractures (discontinuities in the stroke). We divide MNIST training data into clusters and use these transformations to simulate cluster-specific variations. We also add cluster-specific spatial blurring and noise for further variation. 

### Data generation
`generate_morpho_mnist.py` separates MNIST into equally-sized clusters and applies cluster-specific random transformations. This can be run directly to use default parameters, or see `python generate_morpho_mnist.py --help` for specific transformation parameters.

The unmodified, "plain" MNIST dataset must first be downloaded from the [Morpho-MNIST repository](https://github.com/dccastro/Morpho-MNIST). The downloaded zip file should be unzipped into DATADIR/morpho-mnist/plain. These files are needed:

1. train-images-idx3-ubyte.gz: 60k training images
2. train-labels-idx1-ubyte.gz: training labels
3. t10k-images-idx3-ubyte.gz: 10k test images
4. t10k-labels-idx1-ubyte.gz: test labels

### Main script
in progress