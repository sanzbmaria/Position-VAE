# Position_VAE

This Git repository contains code for exploring and preprocessing data, as well as implementing a Variational Autoencoder (VAE) model and visualizing the learned manifold of the model.

## Installation

```
conda create --name vae && conda activate vae && conda config --append channels conda-forge && conda install --file requirements.txt

python3 setup.py install
```

The original data should be stored in src/data/original/json


## Exploration Analysis

```
 python3 src/exploration/stats.py 
 python3 src/exploration/main.py
```

## Preprocessing

To run the preprocessing code use the following command (example):

```
python3 src/preprocessing/main.py
```

## Model 





