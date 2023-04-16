import pytorch_lightning as pl
import os
import shutil
import itertools
import warnings

import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

from umap import UMAP
from torchvision import transforms


## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try:  # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except:  # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def divide_marker(df, level=0):
    """divides the dataframe which contains all markers into a dictionary with
    individual dataframes for each marker."""
    df_dict = {}

    # get the unique markers
    markers = df.columns.get_level_values(level).unique()

    for marker in markers:
        df_dict[marker] = df[marker]
        df_dict[marker] = df[marker].assign(label=marker)
        df_dict[marker]["colFromIndex"] = df[marker].index

    return df_dict


class PlotDiff:
    """Creates plosts to analyze the difference between the original and the reconstructed data"""

    def __init__(self):
        """
        ONLY WORKS FOR 156 VERSION !!
        Args:
            - marker_names (list): list of marker names (from data_preproc)
            - marker_info (list): list of marker info (from data_preproc, eg. ['x', 'y', 'z', 'dis'])
        """
        warnings.warn("This class is only for the 156 version of the data")
        self._marker_names = [
            "Lhand",
            "nose",
            "Lfoot",
            "Lshoulder",
            "RKnee",
            "RShoulder",
            "neck",
            "RFoot",
            "LKnee",
            "tail",
            "RHand",
            "hip",
            "head",
            "distance_barrel",
            "distance_feeder",
        ]
        self._marker_info = ["x", "y", "z", "dis"]
        self._plot_monkey_pose = PlotMonkeyPose()
        # Configure UMAP hyperparameters
        self._reducer = UMAP(
            n_neighbors=15,  # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
            n_components=3,  # default 2, The dimension of the space to embed into.
            metric="euclidean",  # default 'euclidean', The metric to use to compute distances in high dimensional space.
            n_epochs=500,  # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
            learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
            init="spectral",  # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
            min_dist=0.1,  # default 0.1, The effective minimum distance between embedded points.
            spread=0.5,  # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
        )
        self._convert_tensor = transforms.ToTensor()

    def _cluster_plot(self, X):
        # --------------------------------------------------------------------------#
        # This section is not mandatory as its purpose is to sort the data by label
        # so, we can maintain consistent colors for digits across multiple graphs

        # Create a Pandas dataframe using the above array
        df = pd.DataFrame(X, columns=["x", "y", "z"])
        # Convert label data type from float to integer

        # --------------------------------------------------------------------------#

        # Create a 3D graph
        fig = px.scatter_3d(df, x="x", y="y", z="z", height=900, width=950)

        # fig.add_annotation(x=0, y=0, z=0, text="Epoch: {}".format(epoch), showarrow=False)

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

        fig.write_image(f"temp/umap.png", width=500, height=500)

        # open the picture
        img = Image.open(f"temp/umap.png")
        img = self._convert_tensor(img)

        return img

    def cluster_analysis(self, z):
        """Applies UMAP and HDBSCAN to data"""
        # 0. Create temp folder
        if not os.path.exists("temp"):
            os.makedirs("temp")

        z = self._reducer.fit_transform(z.cpu().detach().numpy())
        img = self._cluster_plot(z)
        return img

    def recreation_analysis(self, original, recreated):
        """Compares the original and recreated data
            - pose
            - distance from hip
            - distance from landmarks

        Args:
            - original (tensor) : original data
            - recreated (tensor) : recreated data

        Returns:
            - fig_dic (dictionary): dictionary with plotly images
        """
        # 0. Create temp folder
        if not os.path.exists("temp"):
            os.makedirs("temp")

        assert (
            original.shape == recreated.shape
        ), "The original and recreated data should have the same shape"
        assert original.shape[1] == 156, "The original data should have 156 markers"
        assert recreated.shape[1] == 156, "The recreated data should have 156 markers"

        # 1. Divide the data into pose, distance from hip and distance from landmarks
        original_markers, original_distance, original_landmarks = self.__divide_data(
            original
        )
        recreated_markers, recreated_distance, recreated_landmarks = self.__divide_data(
            recreated
        )

        # 2. Compare the pose
        pose_img = self.__pose_recreation(original_markers, recreated_markers)
        pose_diff = self.__pose_diff(original_markers, recreated_markers)

        # 3. Compare the distance from hip
        hip_img = self.__hip_distance_recreation(original_distance, recreated_distance)

        # 4. Compare the distance from the landmarks
        lmk_img = self.__landmark_distance_recreation(
            original_landmarks, recreated_landmarks
        )

        # 5. remove the temp folder
        shutil.rmtree("temp")

        return {
            "pose_image": pose_img,
            "hip_image": hip_img,
            "pose_diff": pose_diff,
            "lmk_image": lmk_img,
        }

    def __pose_recreation(self, original, recreated):
        """Plots the original and recreated pose to compare
        Args:
            - original (df): dataframe containing the original pose
            - recreated (df): dataframe containing the recreated pose

        Returns:
            - overlap_fig (plotly figure): plotly figure with both ORIGINAL and RECRTEATED
                monkey pose at a single timepoint

        """
        assert (
            original.shape == recreated.shape
        ), "The original and recreated data should have the same shape"

        # choose 10 random indexes
        indexes = np.random.choice(len(original.index.unique()), 10, replace=False)

        # plot the pose
        original_figs = self._plot_monkey_pose(original, "original", indexes=indexes)
        recreated_figs = self._plot_monkey_pose(recreated, "recreated", indexes=indexes)

        # open the pictures
        original_imgs = [Image.open(img) for img in original_figs]
        recreated_imgs = [Image.open(img) for img in recreated_figs]

        # put the images side by side
        images = []

        for oimg, rimg in zip(original_imgs, recreated_imgs):
            images.append(get_concat_h(oimg, rimg))

        # concat the images vertically
        im_v = images[0]

        for i in range(1, len(images)):
            im_v = get_concat_v(im_v, images[i])

        # save the image

        # im_v.save("temp/overlap.png")

        # # open the picture
        # overlap_img = Image.open("temp/overlap.png")

        # convert to tensor
        overlap_img = self._convert_tensor(im_v)

        return overlap_img

    def __pose_diff(self, original, recreated):
        """Calculates the difference between the original and recreated pose
        Args:
            - original (df): dataframe containing the original pose (xyz)
            - recreated (df): dataframe containing the recreated pose (xyz)

        Returns:
            - img (plotly figure): plotly figure with the difference between the original and recreated pose
        """
        original.columns = original.columns.droplevel(1)
        recreated.columns = recreated.columns.droplevel(1)

        # divide the data into x, y and z
        original_x = original.iloc[:, 0::3]
        original_y = original.iloc[:, 1::3]
        original_z = original.iloc[:, 2::3]

        recreated_x = recreated.iloc[:, 0::3]
        recreated_y = recreated.iloc[:, 1::3]
        recreated_z = recreated.iloc[:, 2::3]

        # melt the columns into rows
        original_x = original_x.melt(ignore_index=False).sort_index()
        original_y = original_y.melt(ignore_index=False).sort_index()
        original_z = original_z.melt(ignore_index=False).sort_index()

        recreated_x = recreated_x.melt(ignore_index=False).sort_index()
        recreated_y = recreated_y.melt(ignore_index=False).sort_index()
        recreated_z = recreated_z.melt(ignore_index=False).sort_index()

        # merge the dataframes and calculate the difference
        difference_x = original_x
        difference_x["value"] = original_x["value"].subtract(recreated_x["value"])

        difference_y = original_y
        difference_y["value"] = original_y["value"].subtract(recreated_y["value"])

        difference_z = original_z
        difference_z["value"] = original_z["value"].subtract(recreated_z["value"])

        # add the type column
        difference_x = difference_x.assign(type="x")
        difference_y = difference_y.assign(type="y")
        difference_z = difference_z.assign(type="z")

        # make the value column positive
        difference_x["value"] = difference_x["value"].abs()
        difference_y["value"] = difference_y["value"].abs()
        difference_z["value"] = difference_z["value"].abs()

        # concat the dataframes
        difference = pd.concat([difference_x, difference_y, difference_z])
        difference["index"] = difference.index

        # plot the difference
        fig = px.density_heatmap(
            difference,
            x="index",
            y="variable",
            z="value",
            facet_col="type",
            histfunc="avg",
            title="ABS difference btw xyz",
        )

        fig.write_image(f"temp/xyz_fig.png", width=1000, height=500)

        # open the picture
        fig = Image.open(f"temp/xyz_fig.png")
        fig = self._convert_tensor(fig)
        return fig

    def __landmark_distance_recreation(self, original_lmk, recreated_lmk):
        """Plots the original distances and recreated instances"""

        original_lmk = original_lmk.melt(ignore_index=False)
        recreated_lmk = recreated_lmk.melt(ignore_index=False)

        # sort
        original_lmk = original_lmk.rename_axis("MyIdx").sort_values(
            by=["MyIdx", "variable"], ascending=[True, True]
        )
        recreated_lmk = recreated_lmk.rename_axis("MyIdx").sort_values(
            by=["MyIdx", "variable"], ascending=[True, True]
        )

        # add the type column
        original_lmk = original_lmk.assign(type="original")
        recreated_lmk = recreated_lmk.assign(type="recreated")

        # concat the dataframes
        difference = pd.concat([original_lmk, recreated_lmk]).sort_values(
            by=["MyIdx", "variable"], ascending=[True, True]
        )

        # make the value column positive
        difference["value"] = difference["value"].abs()

        difference["index"] = difference.index

        # plot the difference
        fig = px.density_heatmap(
            difference,
            x="index",
            y="variable",
            z="value",
            histfunc="avg",
            title="ABS difference btw landmark distances",
        )

        fig.write_image(f"temp/distance_lmk_fig.png", width=500, height=500)

        # open the picture
        fig = Image.open(f"temp/distance_lmk_fig.png")
        fig = self._convert_tensor(fig)
        return fig

    def __hip_distance_recreation(self, original, recreated):
        """Plots the original distances and recreated instances
            Args:
            - original (df): dataframe containing the original pose
            - recreated (df): dataframe containing the recreated pose

        Returns:
            - overlap_fig (plotly figure): plotly figure comparing the ORIGINAL and RECRTEATED
                distance from hip and distance from landmarks
        """
        original.columns = original.columns.droplevel(1)
        recreated.columns = recreated.columns.droplevel(1)

        original = original.melt(ignore_index=False)
        recreated = recreated.melt(ignore_index=False)

        original = original.assign(type="original").sort_index()
        recreated = recreated.assign(type="recreated").sort_index()

        # merge the dataframes
        difference = original
        difference["value"] = original["value"].subtract(recreated["value"])
        difference["index"] = difference.index
        difference = difference.drop("type", axis=1)

        # convert the difference['value'] to abs
        difference["value"] = difference["value"].abs()

        fig = px.density_heatmap(
            difference,
            x="index",
            y="variable",
            z="value",
            histfunc="avg",
            title="ABS difference btw dist from hip",
        )

        fig.write_image(f"temp/distance_fig.png", width=500, height=500)

        # open the picture
        fig = Image.open(f"temp/distance_fig.png")
        fig = self._convert_tensor(fig)
        return fig

    def __divide_data(self, data):
        """Divides the data into:
            - pose (xyz)
            - distance from hip
            - distance from markers (barrel and feeder)

        Args:
            data (tensor): tensor used as input

        Returns:
            - markers_df (dataframe): dataframe with the xyz coordinates of the markers
            - distance_df (dataframe): dataframe with the distance from the hip
            - landmark_df (dataframe): dataframe with the distance from the landmarks
        """
        # split data into markers and landmarks
        assert data.shape[1] == 156, "Data is not in the correct shape, should be 156"

        markers, landmark = torch.split(data, [52, 104], dim=1)

        marker_names = self._marker_names

        if "distance_barrel" in marker_names:
            marker_names.remove("distance_barrel")
        if "distance_feeder" in self._marker_names:
            marker_names.remove("distance_feeder")

        # create the xyzdist markers
        col_names = self._marker_info * 13
        marker_num = np.repeat(marker_names, len(self._marker_info))

        # assign to dataframe
        markers_df = pd.DataFrame(markers.numpy(), columns=col_names)
        markers_df = markers_df.T.set_index(marker_num, append=True).T.swaplevel(
            0, 1, 1
        )

        # we want to extract the xyz and leave the dist in another dataframe
        marker_xyz = list(itertools.product(self._marker_names, self._marker_info[:3]))
        marker_dist = list(itertools.product(self._marker_names, self._marker_info[3:]))

        # extract those from original df and divide it
        marker_xyz_df = markers_df.loc[:, marker_xyz]
        marker_dist_df = markers_df.loc[:, marker_dist]

        # make the landmark df
        landmark_df = pd.DataFrame(landmark.numpy())

        return marker_xyz_df, marker_dist_df, landmark_df


class PlotMonkeyPose:
    """Processes the data to plot the monkey pose"""

    def __init__(self):
        self._marker_connections = self.marker_connections
        self._limb_colors = self.limb_colors
        self._joint_colors = self.joint_colors

    @property
    def marker_connections(self):
        """Returns the connections between the markers"""
        self._marker_connections = [
            ["Lfoot", "LKnee"],
            ["LKnee", "hip"],
            ["RFoot", "RKnee"],
            ["RKnee", "hip"],
            ["hip", "neck"],
            ["hip", "tail"],
            ["neck", "nose"],
            ["nose", "head"],
            ["Lhand", "Lshoulder"],
            ["Lshoulder", "neck"],
            ["RHand", "RShoulder"],
            ["RShoulder", "neck"],
        ]

        return self._marker_connections

    @property
    def limb_colors(self):
        """Defines the colors for the limbs"""
        self._limb_colors = {
            "Lfoot-LKnee": "darkviolet",
            "RKnee-hip": "darkviolet",
            "RFoot-RKnee": "darkviolet",
            "LKnee-hip": "darkviolet",
            "RShoulder-neck": "orange",
            "Lhand-Lshoulder": "orange",
            "Lshoulder-neck": "orange",
            "RHand-RShoulder": "orange",
            "hip-neck": "lightgreen",
            "hip-tail": "forestgreen",
            "neck-nose": "lightblue",
            "nose-head": "steelblue",
        }

    @property
    def joint_colors(self):
        """Defines the colors for the joints"""
        self._joint_colors = {
            "nose": "lightcoral",
            "head": "coral",
            "neck": "sienna",
            "RShoulder": "plum",
            "RHand": "tomato",
            "Lshoulder": "darkslateblue",
            "Lhand": "salmon",
            "hip": "seagreen",
            "RKnee": "mediumspringgreen",
            "RFoot": "springgreen",
            "LKnee": "mediumvioletred",
            "Lfoot": "deeppink",
            "tail": "greenyellow",
        }

        return self._joint_colors

    def __call__(self, dataframe, type, indexes):
        """Plots the monkey pose
        Args:
            - dataframe (dataframe): dataframe with the coordinates for the markers at each timepoint
        Returns:
            - fig (plotly figure): plotly figure with the monkey pose at a single timepoint
        """

        # convert to dict
        # divide into individual markers
        dfs_dict = divide_marker(dataframe)
        # create connecting lines
        connecting_lines_df = self._connecting_lines(dfs_dict)

        dataframe = pd.concat(dfs_dict.values())

        # create the plot

        figs = []

        for i in indexes:
            # save the figure
            figs.append(self._plot(dataframe, connecting_lines_df, i, type))

        return figs

    def _connecting_lines(self, dfs_dict):
        """Creates the coordinates for the markers that need to be created to visualize monkey

        Args:
            - dfs_dic ( dictionary) : dictionary containing daframe with colums [x,y,x, label and colFromIndex]

        Returns:
            - connecting_lines_df (dataframe): dataframe with the coordinates for the connecting lines at each timepoint
        """

        c_dict = {}

        for connection in self._marker_connections:
            c_dict[f"{connection[0]}-{connection[1]}"] = pd.concat(
                [dfs_dict[connection[0]], dfs_dict[connection[1]]], axis=1
            ).drop(columns=["label", "colFromIndex"])

        # convert to a list of dataframes
        # todo: why did I do this?
        frames = []
        for key in c_dict.keys():
            temp_b = pd.DataFrame.from_dict(c_dict[key])
            temp_b["labels"] = key
            frames.append(temp_b)

        # convert to pandas
        connecting_lines_df = pd.concat(frames).sort_index()
        connecting_lines_df.columns = ["x1", "y1", "z1", "x2", "y2", "z2", "label"]

        return connecting_lines_df

    def _plot(self, df, con_df, i, type):
        """Helper function to plot the picture at each a single timepoint
        Args:
        - df (dataframe): dataframe with the coordinates for the markers at each timepoint
        - con_df (dataframe): dataframe with the coordinates for the connecting lines at each timepoint
        - i (int): timepoint

        Returns:
         - fig (plotly figure): plotly figure with the monkey pose at a single timepoint

        """

        df = df.loc[[i]]
        con_df = con_df.loc[[i]]

        con_df["x"] = con_df[["x1", "x2"]].values.tolist()
        con_df["y"] = con_df[["y1", "y2"]].values.tolist()
        con_df["z"] = con_df[["z1", "z2"]].values.tolist()

        fig = px.scatter_3d(df, x="x", y="y", z="z", color="label")
        fig.update_traces(marker=dict(size=4))

        for index, row in con_df.iterrows():
            tem = pd.DataFrame(row).T

            x = tem["x"].values[0]
            y = tem["y"].values[0]
            z = tem["z"].values[0]
            label = tem["label"].values[0]

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    name=label,
                    marker={"color": "rgb(0.0, 0.0, 0.0)"},
                    mode="lines",
                )
            )

        fig.update_layout(
            title={
                "text": f"{type}",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(r=20, b=10, l=10, t=10),
            font=dict(
                family="Arial",
                size=18,  # Set the font size here
            ),
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    nticks=6,
                    range=[-3, 3],
                ),
                yaxis=dict(
                    nticks=6,
                    range=[-3, 3],
                ),
                zaxis=dict(
                    nticks=6,
                    range=[-3, 3],
                ),
            ),
            height=500,
            width=500,
        )

        fig.update_layout(showlegend=False)

        fig.write_image(f"temp/monkey_{type}_{i}.png")

        return f"temp/monkey_{type}_{i}.png"


def percentage(percent, number):
    percent = percent / 100
    return round(percent * number)


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
