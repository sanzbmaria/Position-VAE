"""
This module contains functions to preprocess and analyze 3D joint position data from motion capture systems.

Functions:
    - center_hip(df): Centers all the other joints around the hip joint.
    - calculate_distance(df): Calculates the distance between the hip joint and all other joints.
    - check_hip_centered(df): Checks whether the hip joint has been correctly centered in the given DataFrame.
    - add_landmark_location(df): Adds the coordinates of objects (barrels and feeders) to the DataFrame.

"""


import logging as log

from enum import unique

import numpy as np
import pandas as pd
from scipy.sparse import coo
import torch as nn

from absl import flags
from tqdm import tqdm

from scipy.spatial.distance import cdist


FLAGS = flags.FLAGS


def center_hip(df: pd.DataFrame):
    """
    Centers the other joints around the hip

    parameters:
        df (pd.DataFrame): the dataframe to center

    returns:
        df (pd.DataFrame): the dataframe with the hip centered around the rest of the joints
    """

    groups = df.groupby("label")
    hip = groups.get_group("hip")
    hip = hip.drop(columns=["label"])

    temp_df = pd.DataFrame()
    for name, group in groups:
        if name != "hip":
            group = group.drop(columns=["label"])

            group['x'] = group['x'] - hip['x']
            group['y'] = group['y'] - hip['y']
            group['z'] = group['z'] - hip['z']

            group["label"] = name
            temp_df = pd.concat([temp_df, group], axis=0)

    # center the hip
    hip['x'] = 0
    hip['y'] = 0
    hip['z'] = 0

    hip["label"] = "hip"
    temp_df = pd.concat([temp_df, hip], axis=0)

    return temp_df


def calculate_distance(df):
    """
    Calculates the distance between the hip and the rest of the joints

    parameters:
        df (pd.DataFrame|pd.Series): the dataframe to calculate distance for

    returns:
        df (pd.DataFrame):: the dataframe with the calculated distances
    """

    df["distance"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

    return df


def check_hip_centered(df):
    """
    Checks if the hip joint has been centered correctly in the given DataFrame.

    The function groups the data by label, extracts the hip joint data, and then checks if
    the x, y, and z coordinates of the hip joint are all 0. If they are, the hip has been centered correctly.

    Parameters:
        df (pandas.DataFrame| pd.Series): A DataFrame containing joint data with a 'label' column.

    Returns:
        bool (bool): True if the hip joint has been centered correctly, False otherwise.
    """

    groups = df.groupby("label")
    hip = groups.get_group("hip")
    hip = hip.drop(columns=["label"])

    if hip['x'].all() == 0 and hip['y'].all() == 0 and hip['z'].all() == 0:
        return True
    else:
        return False


def add_landmark_location(df: pd.DataFrame):
    """
    Adds the coordinates of the objects (4 barrels and 4 feeders) to the dataframe. 

    Args:
        df (pd.DataFrame): The dataframe to add the object coordinates to

    Returns:
        df (pd.DataFrame): The dataframe with the object coordinates added as columns to each row
    """

    lmks = ['b1', 'b2', 'b3', 'b4', 'f1', 'f2', 'f3', 'f4']

    coords = [
        [1.8831099474237765, 2.2504857710896258, 3.1104950213839317],
        [3.2684966975634477, 2.3323177565826185, 3.0727974914230116],
        [-2.740061407095278, 3.0620530012108063, -2.0537183009440745],
        [-3.090940351965614, 3.0070227700837804, -0.8690008794238412],
        [-2.161960556268887, 2.428982849819521, 3.5207878102359627],
        [3.868664285069503, -0.19594825889453213, 2.5292130833751956],
        [2.5104482176851417, 3.281142061125232, -2.044059905334382],
        [-3.5609489140305444, 1.7744324668075677, -1.8620419355950952]
    ]

    df_lmks = df.copy()

    # calculate distance between each row and each coordinate
    distances = cdist(df[['x', 'y', 'z']], coords)

    # create new columns in dataframe with distances
    for i in range(len(coords)):
        df_lmks[f'distance_to_coord_{lmks[i]}'] = distances[:, i]

    df_lmks.sort_index(inplace=True)

    return df_lmks
