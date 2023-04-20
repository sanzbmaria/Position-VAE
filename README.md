# Position_VAE

This Git repository contains code for exploring and preprocessing data, as well as implementing a Variational Autoencoder (VAE) model and visualizing the learned manifold of the model.

## Documentation
sanzbmaria.github.io/Position_VAE/

# Position_VAE

This Git repository contains code for exploring and preprocessing data, as well as implementing a Variational Autoencoder (VAE) model and visualizing the learned manifold of the model.

## Installation


To install, clone this repository:
```
git clone https://github.com/sanzbmaria/Position_VAE
```

Create and activate a new virtual environment named "vae" using conda, and install the required dependencies using the provided requirements file:
```
conda create --name vae && conda activate vae && conda config --append channels conda-forge && conda install --file requirements.txt
```

## Usage 

Before training the model, the original data should be stored in src/data/original/json, and processed using the following command:
```
python3 src/preprocessing/main.py --log_directory=src/preprocessing/logs --input_directory=src/data/original/ --output_directory=src/data/processed/
```

After processing the data the model can be run using : 

```
python3 src/VAE/main.py -c src/VAE/congfigs/config.yml
```

To visualize the training progress, run the following command:

```
tensorboard --logdir src/VAE/logdir 
```

That's it! You should now be able to explore the learned manifold of the VAE model using the visualizations.
