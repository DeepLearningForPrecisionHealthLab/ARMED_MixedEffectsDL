Synthetic datasets
==================

Setup
-----
Ensure that the root of this repository is on your `PYTHONPATH` environmental
variable. 

Spiral classification
---------------------
Classic two-dimensional nonlinear classification benchmark. Data points are
sampled along two (or more) spiral functions and the neural network must learn
to classify points by their spiral. Random effects are simulated by dividing the
data into clusters and random varying the spiral radius in each cluster. 

See ``synthetic_dataset/spiral_classification.ipynb`` for a comparison of
conventional and mixed effects models. Spiral generation parameters can be
varied to control the degree of random effects and, optionally, confounding
effects.