# Position_VAE

This Git repository contains code for exploring and preprocessing data, as well as implementing a Variational Autoencoder (VAE) model and visualizing the learned manifold of the model.

## Installation

```
conda create --name vae && conda activate vae && conda config --append channels conda-forge && conda install --file requirements.txt

python setup.py install
```

## Exploration Analysis

```
 python3 src/exploration/stats.py --log_directory=src/exploration/logs --input_directory=src/data/original --output_directory=src/data/plots/stats
 python3 src/exploration/main.py --input_directory=src/data --output_directory=src/data/plots --log_directory=src/exploration/logs --log_level=INFO

```

## Preprocessing

To run the preprocessing code use the following command (example):

```
python3 /src/preprocessing/main.py--log_directory=src/preprocessing/logs --input_directory=src/data/original/ --output_directory=src/data/processed
```

## TODO:

- Exploration :
  - monkey video / img
  - Hotspots of the monkey
- Model

- Literature review
