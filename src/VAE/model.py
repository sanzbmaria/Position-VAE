"""
This module defines a Variational Autoencoder (VAE) implemented using PyTorch and PyTorch Lightning.

Classes:
    - Encoder: An encoder network used in the VAE.
    - Decoder: A decoder network used in the VAE.
    - VAE: The main class implementing the VAE.

Example:
    vae = VAE(in_dim=64,  hidden_dims=[32, 16, 8],  latent_dim=2,  learning_rate=1e-3,  batch_size=32,  num_workers=4,  gpus=1,  max_epochs=100,  early_stop_patience=10,  checkpoint_dir='checkpoints',  checkpoint_filename='vae.ckpt',  log_dir='logs',  log_name='vae',  plotter=Plot_NoLandmarks(),  **kwargs)
"""

import torch
import torch.nn as nn

import pytorch_lightning as pl

import plots as plt


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list = None, **kwargs):
        super().__init__()

        layers = []

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        first_layer = in_dim

        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(first_layer, h_dim),
                    nn.Tanh(),
                    nn.Dropout(),
                )
            )
            first_layer = h_dim

            self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int, hidden_dims: list = None,  **kwargs):
        super().__init__()

        layers = []

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        hidden_dims.append(latent_dim)
        hidden_dims.reverse()
        hidden_dims.append(out_dim)
        
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.Dropout(),
                    nn.Tanh(),
                    nn.Linear(hidden_dims[i],hidden_dims[i + 1]),
                )
            )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class VAE(pl.LightningModule):
    def __init__(self, plots, in_dim: int, beta: float = 3, lr: float = 1e-3, hidden_dims: list = None, latent_dim : int = 8, **kwargs):
        super().__init__()
        """
        Instantiate the model with the given hyperparameters
         
        Args:   
            - in_dim: input dimension (52 for no position data, 156 for position data)
            - hidden_dims: list of hidden dimensions 
            - beta (float): beta parameter for the loss function
        """

        self._beta = beta
        self.lr = lr
        self.plots = plots
        self.latent_dim = latent_dim

        self.training_step_xs = []
        self.training_step_xhats = []
        self.training_setp_zs = []

        self.validation_step_xs = []
        self.validation_step_xhats = []
        self.validation_setp_zs = []

        self.validation_lossses = {'MSE': [], 'KLD': [], 'loss': []}
        self.training_lossses = {'MSE': [], 'KLD': [], 'loss': []}

        self.encoder = Encoder(
            in_dim=in_dim, hidden_dims=hidden_dims)

        self.fc_mu = nn.Sequential(
            nn.Linear(
            hidden_dims[-1], latent_dim,
            ),
            nn.Dropout(0.4),)
        self.fc_var = nn.Sequential(
            nn.Linear(
            hidden_dims[-1],latent_dim,
            ),
            nn.Dropout(0.4),)

        self.decoder = Decoder(
            latent_dim = latent_dim, out_dim=in_dim, hidden_dims=hidden_dims)
        
        ## Experimenting 
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):

        # remove the 1 dimensions 
        x_ = x.squeeze(1)

        encoded = self.encoder(x_)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        # reparametrization trick
        z = self.reparametrize(mu, log_var)
        
        x_hat = self.decoder(z)
        
        x_hat_ = x_hat.unsqueeze(1) 

        return [x_hat_, mu, log_var, z]

    def reparametrize(self, mu, log_var):
        # why only mu when not training?

        return mu + log_var* self.N.sample(mu.shape)
    

    def loss_function(self, x, x_hat, mu, log_var, ):
        MSE = nn.functional.mse_loss(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = MSE + self._beta * KLD

        return MSE, KLD, loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = batch
        x_hat, mu, log_var, z = self(x)
        # todo verify the shapes passed to loss function 
        MSE, KLD, loss = self.loss_function(x, x_hat, mu, log_var)

        self.training_lossses['MSE'].append(MSE.detach())
        self.training_lossses['KLD'].append(KLD.detach())
        self.training_lossses['loss'].append(loss.detach())

        self.log("train/MSE", MSE, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)
        self.log("train/KLD", KLD, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):

        # todo: divde by batch size
        avg_loss = torch.stack(self.training_lossses['loss']).mean()
        avg_mse = torch.stack(self.training_lossses['MSE']).mean()
        avg_kdl = torch.stack(self.training_lossses['KLD']).mean()

        # all_preds = torch.stack(self.training_step_outputs)
        # do something with all preds

        self.logger.experiment.add_scalars(
            'epoch/loss', {'train': avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars(
            'epoch/KDL', {'train': avg_kdl}, self.current_epoch)
        self.logger.experiment.add_scalars(
            'epoch/MSE', {'train': avg_mse}, self.current_epoch)

        self.training_step_xs.clear()
        self.training_step_xhats.clear()
        self.training_setp_zs.clear()

        self.training_lossses['MSE'].clear()
        self.training_lossses['KLD'].clear()
        self.training_lossses['loss'].clear()

        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var, z = self(x)
        MSE, KLD, loss = self.loss_function(x, x_hat, mu, log_var)

        # remove the dimensions with size 1
        x = x.squeeze()
        x_hat = x_hat.squeeze()
        z = z.squeeze()
        
        self.validation_lossses['MSE'].append(MSE.detach())
        self.validation_lossses['KLD'].append(KLD.detach())
        self.validation_lossses['loss'].append(loss.detach())

        self.validation_step_xs.append(x.detach())
        self.validation_step_xhats.append(x_hat.detach())
        self.validation_setp_zs.append(z.detach())

        self.log("validation/MSE", MSE, on_step=True,
                 on_epoch=False, prog_bar=False, logger=True)
        self.log("validation/KLD", KLD, on_step=True,
                 on_epoch=False, prog_bar=False, logger=True)
        self.log("validation/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_validation_epoch_end(self):

        # todo: divde by batch size
        avg_loss = torch.stack(self.validation_lossses['loss']).mean()
        avg_mse = torch.stack(self.validation_lossses['MSE']).mean()
        avg_kdl = torch.stack(self.validation_lossses['KLD']).mean()

        self.logger.experiment.add_scalars(
            'epoch/loss', {'valid': avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars(
            'epoch/KDL', {'valid': avg_kdl}, self.current_epoch)
        self.logger.experiment.add_scalars(
            'epoch/MSE', {'valid': avg_mse}, self.current_epoch)

        all_xs = torch.stack(self.validation_step_xs)
        all_xhats = torch.stack(self.validation_step_xhats)
        all_zs = torch.stack(self.validation_setp_zs)

        # join the first two dimensions
        all_xs = all_xs.view(-1, all_xs.shape[-1])
        all_xhats = all_xhats.view(-1, all_xhats.shape[-1])
        all_zs = all_zs.view(-1, all_zs.shape[-1])

        # clear the lists
        self.validation_step_xs.clear()
        self.validation_step_xhats.clear()
        self.validation_setp_zs.clear()

        self.validation_lossses['MSE'].clear()
        self.validation_lossses['KLD'].clear()
        self.validation_lossses['loss'].clear()

        plots_dict = self.plots(x=all_xs, xhat=all_xhats,
                                z=all_zs, epoch=self.current_epoch, n=10)
        for key in plots_dict:
            self.logger.experiment.add_image(
                f"Reconstruction/{key}",  plots_dict[key], self.current_epoch)

        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.lr))

        # return {"optimizer": optimizer, "monitor": "validation/loss"}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation/loss_epoch"}
