"""
Parent Class that provides methods for visualizing the original and reconstructed data for the model.

Args:
    data_path (str): The path to the directory where the plots will be saved. Default is 'None'.
    label_path (str): The path to the file containing the joint labels. Default is 'None'.
    min_cluster_size (int): The minimum number of samples in a cluster. Default is 10.
    umap_interval (int): The interval for applying UMAP to the data. Default is 5.
    umap_input (int): The number of input samples for UMAP. Default is 1000.

"""

import os
import io
import re
import time
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import hdbscan
import umap 

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import axis3d
import matplotlib.pyplot as plt

from abc import abstractmethod

import torch
from torch import tensor
import torchvision.transforms as transforms

from PIL import Image
from umap import UMAP

import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objs.scattermapbox as smb
import plotly.io as pio
import plotly.subplots as sp


class Plot:
    def __init__(self, data_path: str = 'None', label_path: str = 'None', min_cluster_size: int = 10, umap_interval : int = 5, umap_input: int = 1000) -> None:
        """
        Initializes the Plot class.

        Args:
            data_path (str): The path to the directory where the plots will be saved. Default is 'None'.
            label_path (str): The path to the file containing the joint labels. Default is 'None'.
            min_cluster_size (int): The minimum number of samples in a cluster. Default is 10.
            umap_interval (int): The interval for applying UMAP to the data. Default is 5.
            umap_input (int): The number of input samples for UMAP. Default is 1000.
        """
        
        self._data_path = data_path
        self._label_path = label_path
        self.umap_interval = umap_interval 
        self.num_joints = 13
        self.umap_input = umap_input
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

        # check if the datapath exists if not create it
        self._data_path = os.path.join(self._data_path, 'plots')

        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)
            
            
        # load the labels
        with open(self._label_path, 'rb') as f:
            self.joint_labels = pickle.load(f)
        
        if self.num_joints == 13:
            self.joint_colors = {i: color for i, color in enumerate(["lightcoral", "coral", "sienna", "plum", "tomato", "darkslateblue", "salmon", "seagreen", "mediumspringgreen", "springgreen", "mediumvioletred", "deeppink", "greenyellow"])}
            self.joint_graph = [["neck","head"],["neck","RShoulder"],["neck","Lshoulder"],["neck","hip"],["head","nose"],["RShoulder","RHand"],["Lshoulder","Lhand"],["hip","RKnee"],["hip","LKnee"],["hip","tail"],["LKnee","Lfoot"],["RKnee","RFoot"]]
        else:
            raise ValueError("Number of joints not supported")
        
        self.umap_reducer = UMAP(
            n_neighbors=15,  # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
            n_components=3,  # default 2, The dimension of the space to embed into.
            metric="euclidean",  # default 'euclidean', The metric to use to compute distances in high dimensional space.
            n_epochs=3,  # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
            learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
            init="spectral",  # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
            min_dist=0.1,  # default 0.1, The effective minimum distance between embedded points.
            spread=1.0,  # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
        )


        self.hdb = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
            gen_min_span_tree=False, leaf_size=100,
            metric='euclidean', min_samples=None, p=None, min_cluster_size=10)
        
    @abstractmethod
    def __call__(self, **kwargs) -> None:
        """
        Abstract method for plotting.
        """
        raise NotImplementedError
    
    @abstractmethod
    def divide_tensor(self, **kwargs):
        """
        Divides the tensor into the different parts (xyz, distance, landmark distance, etc)
        """
        raise NotImplementedError
    
    def plot_landmark_distance_heatmap(self, x_lmk_dist, xhat_lmk_dist):
        """
        Plots the heatmap of the landmark distance. 
        """
        raise NotImplementedError
    
       
    def recreation_analysis(self, x_xyz : pd.DataFrame,
                            xhat_xyz: pd.DataFrame,  
                            x_dist: pd.DataFrame, 
                            xhat_dist: pd.DataFrame,
                            x_lmk_dist=None,
                            xhat_lmk_dist=None, 
                            z=None,
                            n_timepoints=2, 
                            current_epoch=0):
        """
        Compares the original data with the reconstructed data
        
        Args:
            x_xyz (Pandas.df): original data xyz values
            xhat_xyz (Pandas.df): reconstructed data xyz values
            x_dist (Pandas.df):  original data distance from hip values  
            xhat_dist (Pandas.df): reconstructed data distance from hip values
            x_lmk_dist (Pandas.df): (optional: only for Landmark Version) original data distance from landmarks values
            xhat_lmk_dist (Pandas.df): (optional: only for Landmark Version) reconstructed data distance from landmarks values
            z (torch.tensor): (optional: only for Landmark Version) tensor of the model's encoded latent variables 
            n_timepoints (int): number of random samples to compare (pose only)
            current_epoch (int): current epoch
        
        Returns: 
            fig_dic (dict): dictionary with tensors of the figures
        """
        
        # choose n random indexes to plot 
        if n_timepoints > len(x_xyz.index.unique()):
            n_timepoints = len(x_xyz.index.unique())
        
        indexes = np.random.choice(len(x_xyz.index.unique()), n_timepoints, replace=False)

        # plot the heatmaps
        
        xyz_diff = self.calculate_differences(x_xyz, xhat_xyz)
        hip_dist_diff = self.calculate_differences(x_dist, xhat_dist)
        
        heatmaps_xyz_fig = self.plot_xyz_heatmap(xyz_diff)
        heatmap_dist_fig = self.plot_hip_distance_heatmap(hip_dist_diff)
        
        heatmap_dist_lmk_fig = None
        
        if x_lmk_dist is not None:
            # todo ! 
            # lmk_dist_diff = self.calculate_differences(x_lmk_dist, xhat_lmk_dist)
            # heatmap_dist_lmk_fig = self.plot_landmark_distance_heatmap(lmk_dist_diff)
            heatmap_dist_lmk_fig = None
        
        # close the figures
        matplotlib.pyplot.close()
        
        # plot the joints in 3D space 
        joints_fig = self.plot_joints(original_df = x_xyz,recreated_df= xhat_xyz, indexes= indexes)
                
        # close the figures
        matplotlib.pyplot.close()
                
        if z is not None and (current_epoch % self.umap_interval  == 1 or current_epoch == 1):
            # plot the clusters using umap
            cluster_fig = self.cluster_analysis(z)

            # close the figures
            matplotlib.pyplot.close()
            
            # join the dictionarys into one
            fig_dict = {**heatmaps_xyz_fig, **heatmap_dist_fig, **heatmap_dist_lmk_fig, **joints_fig, **cluster_fig} if heatmap_dist_lmk_fig is not None else {**heatmaps_xyz_fig, **heatmap_dist_fig, **joints_fig, **cluster_fig}
        else:
            fig_dict = {**heatmaps_xyz_fig, **heatmap_dist_fig, **heatmap_dist_lmk_fig, **joints_fig} if heatmap_dist_lmk_fig is not None else {**heatmaps_xyz_fig, **heatmap_dist_fig, **joints_fig}
            
        tensor_dict = {}
        
        for fig in fig_dict:
            # open the picture and convert it to tensor
            
            img = Image.open(fig_dict[fig])  
            tensor_img = self.transform(img)
            tensor_dict[fig] = tensor_img

        return tensor_dict


    def error_distribution_plot(self, x : torch.tensor , xhat : torch.tensor):
        """
        Plots a histogram of the error between the original and recreated data.

        Args:
            x (torch.Tensor): A tensor of shape (num_samples, 52/156) representing the original data.
            xhat (torch.Tensor): A tensor of shape (num_samples, 52/156) representing the recreated data.

        Returns:
            dict (dic): A dictionary containing the name of the figure and the path to the figure.
        """
        
        x = torch.reshape(x, (x.shape[0],  self.num_joints, -1))
        xhat = torch.reshape(xhat, (xhat.shape[0],  self.num_joints, -1))


        # Calculate the mean squared error (MSE) or mean absolute error (MAE) between the original and recreated data
        errors = torch.mean((x - xhat) ** 2, dim=(1, 2))  # for MSE
        # errors = torch.mean(torch.abs(original_data - recreated_data), dim=(1, 2))  # for MAE

        fig, ax = plt.subplots()
        ax.hist(errors.cpu().numpy(), bins=30, density=True, alpha=0.75)
        ax.set_xlabel('Error Value')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution between Original and Recreated Data')
        
        # save the figure
        path = os.path.join(self._data_path, 'error_distribution_fig.png')
        fig.savefig(path)

        img = Image.open(path)  
        tensor_img = self.transform(img)
    
        return {'error_distribution_fig': tensor_img}
        

    def cluster_plot(self, X: np.array, y :np.array = None):
        """
        Plots a 3D scatter plot of the input data X with consistent colors for the same label.

        Args:
            X (numpy.array): ndarray The input data to plot in 3D. It must have at least three columns.
            y (numpy.array): ndarray The labels for the input data. If None, the labels will be generated automatically. (optional)

        Returns:
            str (str): The path to the saved image.
        """


        if y is not None:
             # Convert label data type from float to integer
            arr_concat=np.concatenate((X, y.reshape(y.shape[0],1)), axis=1)
            # Create a Pandas dataframe using the above array
            df=pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
            # Convert label data type from float to integer
            df['label'] = df['label'].astype(int)
            # Finally, sort the dataframe by label
            df.sort_values(by='label', axis=0, ascending=True, inplace=True)
            
            # Create a 3D graph
            fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), height=900, width=950, color_discrete_sequence=px.colors.qualitative.Vivid)

        else:
            # Create a Pandas dataframe using the above array
            df = pd.DataFrame(X, columns=["x", "y", "z"])

            # Create a 3D graph
            fig = px.scatter_3d(df, x="x", y="y", z="z", height=900, width=950)


        # Update chart looks
        fig.update_layout(
            title_text="{type}",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.5, y=-1.4, z=0.5),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(
                    backgroundcolor="white",
                    color="black",
                    gridcolor="#f0f0f0",
                    title_font=dict(size=10),
                    tickfont=dict(size=10),
                ),
                yaxis=dict(
                    backgroundcolor="white",
                    color="black",
                    gridcolor="#f0f0f0",
                    title_font=dict(size=10),
                    tickfont=dict(size=10),
                ),
                zaxis=dict(
                    backgroundcolor="lightgrey",
                    color="black",
                    gridcolor="#f0f0f0",
                    title_font=dict(size=10),
                    tickfont=dict(size=10),
                ),
            ),
        )
        # Update marker size
        fig.update_traces(marker=dict(size=3, line=dict(color="black", width=0.1)))

        path = f"{self._data_path}/cluster_img.png"
        fig.write_image(path)
        
        return path

    def cluster_analysis(self, z : torch.tensor):
        """
        Applies the UMAP dimensionality reduction technique and HDBSCAN clustering algorithm to input data z, and
        plots a 3D scatter plot of the reduced data with consistent colors for the same label.

        Args:
            z (torch.tensor): tensor The input data to cluster. It must be a PyTorch tensor.

        Returns:
            dict (dict): A dictionary containing the path to the image file.
        """ 
        
        #if z is too large, reduce it to n samples
        if z.shape[0] > self.umap_input:
            # choose 1000 random samples
            z = z[np.random.choice(z.shape[0], self.umap_input, replace=False), :]
        
        #print('clustering...')

        z_umap = self.umap_reducer.fit_transform(z.detach().numpy())
        
        # Apply HDBSCAN
        clusterer = self.hdb.fit(z_umap)
        
        # get the number of clusters 
        res = np.array(clusterer.labels_) 
        unique_res = np.unique(res) 

        # make a color palette with seaborn.
        color_palette = sns.color_palette('flare', len(unique_res))


        cluster_colors = [color_palette[x] if x >= 0
                        else (0.5, 0.5, 0.5)
                        for x in clusterer.labels_]

        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                zip(cluster_colors, clusterer.probabilities_)]


        path = self.cluster_plot(z_umap, clusterer.labels_)
        
        return {'UMAP_img': path}
    
    def calculate_differences(self, x: pd.DataFrame, xhat:pd.DataFrame):
        """
        Calculates the differences between two Pandas DataFrames `x` and `xhat`.

        Args:
            x (pd.DataFrame): A DataFrame of shape (num_samples, 52) representing the original data.
            xhat (pd.DataFrame): A DataFrame of shape (num_samples, 52) representing the reconstructed data.

        Returns:
            (pd.DataFrame): A DataFrame containing the median differences between the original and reconstructed data, grouped by labels and bins.
        """

        # Make a copy of xhat
        xhat_0 = xhat.copy()
        xhat_0['joint'] = 0

        diff = x - xhat_0 
        diff = diff.sort_values(by=[diff.index.name, 'joint'])

        # Set the labels based on the joint number
        diff['labels'] = diff['joint'].map(self.joint_labels)

        # Bin the data based on the 'index'
        bins = pd.cut(diff.index, bins=20)
        diff['bins'] = bins

        diff = diff.groupby(['labels', 'bins']).median().reset_index()

        return diff
    
    def plot_xyz_heatmap(self, diff: pd.DataFrame):
        """
        Plots separate heatmaps for the differences in x, y, and z coordinates.

        Args:
            diff (pd.DataFrame): A DataFrame containing the median differences between the original and reconstructed data, grouped by labels and bins.

        Returns:
            dict (dict): A dictionary containing the path to the image file.
        """

        columns = ["x", "y", "z"]
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        for i, col in enumerate(columns):
            heatmap_data = diff.pivot_table(index="labels", columns="bins", values=col)
            im = axes[i].imshow(heatmap_data, cmap="viridis")
            axes[i].set_title(f"Difference btw {col.upper()} (Median)")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Joint")
            plt.colorbar(im, ax=axes[i], shrink=0.6)

        # Save the plot as a PNG file
        path = f"{self._data_path}/xyz_recreation_img.png"

        plt.tight_layout()
        plt.savefig(path)

        return {'xyz_diff_img': path}

        
    def plot_hip_distance_heatmap(self, diff):
        """
        Plots a heatmap for the differences in distances between each joint.

        Args:
            diff (pd.DataFrame): A DataFrame containing the median differences between the original and reconstructed data, grouped by labels and bins.

        Returns:
            dict (dict): A dictionary containing the path to the image file.
        """
    
        heatmap_data = diff.pivot_table(index="labels", columns="bins", values="dist_hip")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_data, cmap="viridis")
        ax.set_title("Difference btw Distances (Median)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Joint")
        plt.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()

        # Save the figure
        path = f"{self._data_path}/dist_recreation_img.png"

        plt.savefig(path)

        return {'dist_diff_img': path}
    
    def plot_joints(self, original_df, recreated_df, indexes):
        """
        Create a side-by-side comparison of 3D plots for each index with its associated joints
        using the original and recreated dataframes.

        Args:
            original_df (pandas DataFrame): The original dataframe with columns [index, joint, x, y, z]
            recreated_df (pandas DataFrame): The recreated dataframe with columns [index, joint, x, y, z]
            indexes (list): A list of indexes to plot
            
        Returns:
            dict (dict) : A dictionary containing the path to the image file.
        """
        
        # todo: add labels to the scatter plot
        # todo: if there are landmarks, add them to the plot ? 
        
        n_indexes = len(indexes)
        
        # Change the labels of the joints to match the ones in the original dataset
        original_df['label'] = original_df['joint'].map(self.joint_labels)
        recreated_df['label'] = recreated_df['joint'].map(self.joint_labels)
        
        original_df['color'] = original_df['joint'].map(self.joint_colors)
        recreated_df['color'] = recreated_df['joint'].map(self.joint_colors)

        indexes_list = list(set(indexes.tolist())) 
        
        original_df = original_df.loc[indexes_list] 
        recreated_df = recreated_df.loc[indexes_list]
        
        # Make index a column
        original_df['index'] = original_df.index
        recreated_df['index'] = recreated_df.index
        
        # Group the data by index
        original_grouped_index = original_df.groupby('index')
        recreated_grouped_index = recreated_df.groupby('index')
        
        
        # Create a figure with n_timepoints rows and one column of subplots
        fig, axs = plt.subplots(n_indexes, 2, sharex=False, subplot_kw={'projection': '3d'}, figsize=(10, 5 * n_indexes))


        # Iterate over each group and create a 3D plot
        for row, (grouped_orig, grouped_recr) in enumerate(zip(original_grouped_index, recreated_grouped_index)):
            # Plot each joint
                 
            jnt_data_orig = grouped_orig[1]
            jnt_data_recr = grouped_recr[1]
            
            axs[row][0].scatter(jnt_data_orig['x'], jnt_data_orig['y'], jnt_data_orig['z'], label='original' , c=jnt_data_orig['color'])
            axs[row][1].scatter(jnt_data_recr['x'], jnt_data_recr['y'], jnt_data_recr['z'], label='recreated', c=jnt_data_orig['color'])

            
            axs[row][0].set_title('Timepoint ' + str(row+1) + ' Original')
            axs[row][1].set_title('Recreated ' + str(row+1))
        
                            
              # Plot lines between connected joints using joint_graph
            for edge in self.joint_graph:
                # Get coordinates of connected joints for the original data
                joint1_orig = jnt_data_orig.loc[jnt_data_orig['label'] == edge[0]]
                joint2_orig = jnt_data_orig.loc[jnt_data_orig['label'] == edge[1]]

                # Get coordinates of connected joints for the recreated data
                joint1_recr = jnt_data_recr.loc[jnt_data_recr['label'] == edge[0]]
                joint2_recr = jnt_data_recr.loc[jnt_data_recr['label'] == edge[1]]

                # Plot lines between connected joints in the original data
                axs[row][0].plot([joint1_orig['x'], joint2_orig['x']],
                                [joint1_orig['y'], joint2_orig['y']],
                                [joint1_orig['z'], joint2_orig['z']],
                                c='k', linestyle='-', linewidth=0.8)

                # Plot lines between connected joints in the recreated data
                axs[row][1].plot([joint1_recr['x'], joint2_recr['x']],
                                [joint1_recr['y'], joint2_recr['y']],
                                [joint1_recr['z'], joint2_recr['z']],
                                c='k', linestyle='-', linewidth=0.8)


                # remove the extra space between subplots
        fig.tight_layout()
        
        # save the figure
        path = f"{self._data_path}/3d_joints_img.png"
        fig.savefig(path)
        
        return {'3d_img': path}