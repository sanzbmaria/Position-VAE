# Position_VAE

This Git repository contains code for exploring and preprocessing data, as well as implementing a Variational Autoencoder (VAE) model and visualizing the learned manifold of the model.

## Installation

```
python setup.py install
```

## Preprocessing

To run the preprocessing code use the following command (example):

```
python3 exploration_analysis/main.py --log_directory=logs --input_directory=../data/original/ --output_directory=../data/processed/
```

## Exploration Analysis

```
    python3 exploration/main.py --input_directory=data --output_directory=data/plots --log_directory=exploration/logs --log_level=INFO

```

## TODO:

- Exploration :
  - monkey video / img
  - Hotspots of the monkey
- Model

- Literature review
