"""Child Class that provides methods for visualizing the original and reconstructed data for the model for the LANDMARKS data.

Args:
    data_path (str): The path to the directory where the plots will be saved. Default is 'None'.
    label_path (str): The path to the file containing the joint labels. Default is 'None'.
    min_cluster_size (int): The minimum number of samples in a cluster. Default is 10.
    umap_interval (int): The interval for applying UMAP to the data. Default is 5.
    umap_input (int): The number of input samples for UMAP. Default is 1000.

"""

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axis3d
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import tensor
import torchvision.transforms as transforms
from PIL import Image

import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objs.scattermapbox as smb
import plotly.io as pio
import plotly.subplots as sp

from .plot_parent import Plot

class Plot_Landmarks(Plot):
    def __init__(self, data_path: str = 'None', label_path: str = 'None', min_cluster_size: int = 10, umap_interval : int = 5, umap_input: int = 1000, **kwargs):
            super().__init__(data_path, label_path, min_cluster_size, umap_interval, umap_input)
        
    def __call__(self, x: torch.tensor, xhat: torch.tensor, z:torch.tensor,  epoch: int, n : int = 10, **kwargs):
        ###### NO LANDMARKS ######

        assert x.shape[1] == 156, "The input data should have 156 features"
        assert xhat.shape[1] == 156, "The output data should have 156 features"

        x = x.reshape(-1, 156).to('cpu')
        xhat = xhat.reshape(-1, 156).to('cpu')
        z = z.squeeze(dim=1).to('cpu')
    
        x_xyz_df, x_dist_hip_df, x_dist_lmks_df = self.divide_tensor(x)
        xhat_xyz_df, xhat_dist_hip_df, xhat_dist_lmks_df = self.divide_tensor(xhat)
        
        error_img = self.error_distribution_plot(x, xhat)
        recreation_imgs = self.recreation_analysis(
            x_xyz = x_xyz_df, 
            x_dist = x_dist_hip_df, 
            xhat_xyz = xhat_xyz_df,
            xhat_dist = xhat_dist_hip_df, 
            x_lmk_dist = x_dist_lmks_df,   
            xhat_lmk_dist = xhat_dist_lmks_df,  
            z =z, 
            n_timepoints = n, 
            current_epoch = epoch)
        
        # join the dicts
        imgs_dict = {**error_img, **recreation_imgs}

        return imgs_dict
    
    
        
    def divide_tensor(self, tensor):
        """
        Divides a given tensor into separate dataframes for xyz values and distance from hip values for each joint.

        Args:
            tensor (torch.tensor): A tensor of shape(batch_size, 13, 12), where the 13x12 tensor has the following structure: [joint_1_x, joint_1_y, joint_1_z, joint_1_dist_hip, ...,joint_13_x, joint_13_y, joint_13_z, joint_13_dist_hip]

        Returns:
            xyz_df(pandas dataframe): A dataframe with columns 'joint', 'x', 'y', and 'z', where each row corresponds to a joint and the xyz values for that joint.
            dist_hip_df (pandas dataframe): A dataframe with columns 'joint' and 'dist', where each row corresponds to a joint and the distance from hip value for that joint.
            df_landmarks (pandas dataframe): A dataframe with columns 'joint', 'dist_b1', 'dist_b2', 'dist_b3', 'dist_b4', 'dist_f1', 'dist_f2', 'dist_f3', 'dist_f4', where each row corresponds to a joint
        """
        
        tensor = tensor.reshape(-1, 13, 12)
        # reshape the tensor into a 2D array with shape (batch * 13, 4)
        reshaped_tensor = tensor.reshape(-1, 12)

        df = pd.DataFrame(reshaped_tensor, 
                          columns=['x', 'y', 'z', 
                                   'dist_b1', 'dist_b2', 'dist_b3', 'dist_b4', 
                                   'dist_f1', 'dist_f2', 'dist_f3', 'dist_f4',  
                                   'dist_hip'])

        # create a 'batch' column using np.indices and reshape to 1D array
        batch_col = np.indices((tensor.shape[0], tensor.shape[1]))[0].reshape(-1)

        df['joint'] = df.index % 13

        # add the 'batch' column to the DataFrame
        df['batch'] = batch_col

        df.set_index('batch', inplace=True)

        df.index.name = ''

        # create a DataFrame with columns 'x', 'y', 'z', and 'joint'
        df_xyz_joint = df.loc[:, ['x', 'y', 'z', 'joint']]

        # create a DataFrame with columns 'joint' and 'dist_hip'
        df_joint_dist_hip = df.loc[:, ['joint', 'dist_hip']]
        
        df_landmarks = df.loc[:, ['joint', 'dist_b1', 'dist_b2', 'dist_b3', 'dist_b4', 'dist_f1', 'dist_f2', 'dist_f3', 'dist_f4']]

        return df_xyz_joint, df_joint_dist_hip, df_landmarks

    def plot_landmark_distance_heatmap(self, diff):
        # todo: plot the heatmap of the distance from landmarks
        raise NotImplementedError