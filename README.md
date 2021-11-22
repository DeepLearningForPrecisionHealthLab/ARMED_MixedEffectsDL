# Mixed Effects Deep Learning

## Setup 
Global directory paths should be editted in `medl.settings`:

1. `RESULTSDIR`: where experimental results will be stored
2. `DATADIR`: where downloaded and simulated datasets are stored

Add the repository root to the `PYTHONPATH`. If using Visual Studio Code, this can be done by modifying the `.env` file, which is read by the Python extension when running any code interactively. 

## Table of contents

* [Synthetic datasets](./synthetic_dataset): spiral classification and MNIST simulations with random effects
* [MCI conversion](./ad_conversion): classification of stable vs. progressive mild cognitive impairment