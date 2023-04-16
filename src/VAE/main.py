""""Main file for VAE training and testing."""

import os
import yaml

import pytorch_lightning as pl


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelCheckpoint



from pytorch_lightning.accelerators import CUDAAccelerator
from model import VAE
from data import VAEDataset

import torch

from plots.plot52 import Plot_NoLandmarks
from plots.plot156 import Plot_Landmarks


def main():
    # Parse arguments YAML file

    with open('src/VAE/configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
        

    torch.set_float32_matmul_precision('medium')

    logger = TensorBoardLogger(
        save_dir=config['tensorboard']['logdir'], name='', sub_dir=config['tensorboard']['name'])
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

        
    # select the correct plot class
    if config['tensorboard']['name'] == 'no_landmarks':
        plot = Plot_NoLandmarks(data_path=config['tensorboard']['logdir'], label_path=config['plots']['label_path'],min_cluster_size= config['plots']['min_cluster_size'], umap_interval = config['plots']['umap_interval'], umap_input=config['plots']['umap_input'] )
    else :
        plot = Plot_Landmarks(data_path=config['tensorboard']['logdir'], label_path=config['plots']['label_path'],min_cluster_size= config['plots']['min_cluster_size'], umap_interval = config['plots']['umap_interval'], umap_input=config['plots']['umap_input'] )
    
    # --------------------------------
    # Step 1: Define a LightningModule
    # --------------------------------
    # A LightningModule (nn.Module subclass) defines a full *system*
    # (ie: an LLM, difussion model, autoencoder, or simple image classifier).

        
    model = VAE(in_dim=config['model']['in_dim'], hidden_dims=config['model']
                ['hidden_dims'], beta=config['model']['beta'], plots=plot)

    # -------------------
    # Step 2: Define data
    # -------------------
    # Define a dataset (or dataloader) for training and testing.

    data = VAEDataset( data_path=config['data']['dir'], 
        **config["data"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
        )
    data.setup()

    # -------------------
    # Step 3: Train
    # -------------------
    
    checkpoint_path = os.path.join(logger.log_dir, 'checkpoint')
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor="validation/loss_epoch")
    earlystopping_callback = EarlyStopping(monitor = "validation/loss_epoch", min_delta=0.0, patience=10, mode='min')
    # lr_finder_callback = LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=1000, mode='exponential', early_stop_threshold=4.0)

    trainer = Trainer(logger=logger, accelerator=CUDAAccelerator(), auto_lr_find=True, callbacks=[
                      lr_monitor, checkpoint_callback, earlystopping_callback],  profiler="simple", **config['trainer_params'])

    trainer.tune(model, datamodule=data)
    #print("RL", model.lr)
    
    trainer.fit(model, datamodule=data)
    
    # save the config file to the log directory
        
    path = logger.log_dir + '/config.yml'
    with open(path, 'w') as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    main()
