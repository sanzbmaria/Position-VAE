import json
import logging as log
import os
import random

import numpy as np
import pandas as pd
import torch as nn
from absl import flags

from utils import setup_utils


def replace_outliers(df, coordinate="z"):
    """
    Replaces outliers in a pandas DataFrame with the previous or next value that is not an outlier.

    Parameters:
        df (pandas.DataFrame): The input DataFrame with two columns, 'value' and 'is_outlier'.

    Returns:
        pandas.DataFrame: The modified DataFrame with outliers replaced.
    """
    # Create a new column called 'value2' that has the same values as 'value'
    df[f"{coordinate}_new"] = df[f"{coordinate}"]

    # if 'outlier_{coordinate}' == 1 replace with NaN
    df.loc[df[f"outlier_{coordinate}"] == 1, f"{coordinate}_new"] = np.nan

    # Forward fill NaNs with the next non-outlier value
    df[f"{coordinate}_new"] = df[f"{coordinate}_new"].fillna(method="ffill")

    # Backward fill NaNs with the previous non-outlier value
    df[f"{coordinate}_new"] = df[f"{coordinate}_new"].fillna(method="bfill")

    # Drop the f'{coordinate}' column and rename 'value2' to 'value'
    df = df.drop(f"{coordinate}", axis=1).rename(
        columns={f"{coordinate}_new": f"{coordinate}"}
    )
    return df


def iqr_method(group_data, label, replace=True):
    """interpolates the data using the IQR method

    Parameters:
        group_data (pandas.DataFrame): The input DataFrame with three columns, 'x', 'y', 'z'
        label (str): The label of the group aka the joint
        replace (bool): If True, replaces the outliers with the previous or next value that is not an outlier.

    Returns:
        pandas.DataFrame: The modified DataFrame with outliers replaced. (if replace=True)
        stats (dict): The stats of the group_data
    """

    # check if the group_data includes label
    if "label" in group_data.columns:
        # drop the label column
        group_data = group_data.drop(["label"], axis=1)

    Q1 = group_data.quantile(0.25)
    Q3 = group_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # get the stats of each coordinate
    x_stats = group_data["x"].describe().to_dict()
    y_stats = group_data["y"].describe().to_dict()
    z_stats = group_data["z"].describe().to_dict()

    # create a column and mark the outliers
    group_data["outlier_x"] = 0
    group_data.loc[group_data["x"] < lower_bound["x"], "outlier_x"] = 1
    group_data.loc[group_data["x"] > upper_bound["x"], "outlier_x"] = 1

    group_data["outlier_y"] = 0
    group_data.loc[group_data["y"] < lower_bound["y"], "outlier_y"] = 1
    group_data.loc[group_data["y"] > upper_bound["y"], "outlier_y"] = 1

    group_data["outlier_z"] = 0
    group_data.loc[group_data["z"] < lower_bound["z"], "outlier_z"] = 1
    group_data.loc[group_data["z"] > upper_bound["z"], "outlier_z"] = 1

    # send the outliers to stat
    outlier_x = group_data[group_data["outlier_x"] == 1]
    outlier_y = group_data[group_data["outlier_y"] == 1]
    outlier_z = group_data[group_data["outlier_z"] == 1]

    # save the stats to a dict
    stats = {
        "percentages": {
            "outlier_x": len(outlier_x) / len(group_data),
            "outlier_y": len(outlier_y) / len(group_data),
            "outlier_z": len(outlier_z) / len(group_data),
            "label": label,
        },
        "x_stats": x_stats,
        "y_stats": y_stats,
        "z_stats": z_stats,
    }

    df = group_data.copy()

    if replace:
        # interpolate the outliers
        # Replace outliers in the x column
        df = replace_outliers(df, "x")

        # Replace outliers in the y column
        df = replace_outliers(df, "y")

        # Replace outliers in the z column
        df = replace_outliers(df, "z")

        # remove outlier column
        df = df.drop(["outlier_x", "outlier_y", "outlier_z"], axis=1)

        log.info(f"Done interpolating outliers for label: {label}")
        
    

    return df, stats, group_data


def interpolate(dataframe):
    """interpolates the data to make it more uniform.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame with four columns, 'x', 'y', 'z', and 'label'.

    Returns:
            pandas.DataFrame: The modified DataFrame with outliers replaced."""

    # order by index
    dataframe = dataframe.sort_index()

    # group by label
    groups = dataframe.groupby("label")

    # interpolate
    new_dataframe = pd.DataFrame()
    new_df_outliers = pd.DataFrame()

    for name, group in groups:
        # interpolate
        group = group.drop(columns=["label"])
        group, stats, df_outliers = iqr_method(group, name, replace=True)

        group["label"] = name
        df_outliers["label"] = name
        
        new_dataframe = pd.concat([new_dataframe, group])
        new_df_outliers = pd.concat([new_df_outliers, df_outliers])
        
    # Removing rows with outlier values
    outliers = ['outlier_x', 'outlier_y', 'outlier_z']
    outlier_indices = new_df_outliers[new_df_outliers[outliers].any(axis=1)].index
    
    new_df_outliers = new_df_outliers.drop(outlier_indices)
    new_df_outliers = new_df_outliers.drop(['outlier_x', 'outlier_y', 'outlier_z'], axis=1)
    
    return new_dataframe, new_df_outliers

            
def get_stats(dataframe, mapping):
    """Calls the IQR method to get the stats of the data for each label

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame with four columns, 'x', 'y', 'z', and 'label'.
        mapping (dict): The mapping of the labels to the joints {key: label, value: number}
    Returns:
            dict: The stats of the data for each label.
            new_df: The df containing info of whether or not the timepoint is an outlier """

    # order by index
    dataframe = dataframe.sort_index()

    # group by label
    groups = dataframe.groupby("label")

    new_dataframe = pd.DataFrame()
    all_stats = {}

    for name, group in groups:
        # interpolate
        group = group.drop(columns=["label"])
        group, stats = iqr_method(group, mapping[name], replace=False)

        group["label"] = name

        all_stats[mapping[name]] = stats
        new_dataframe = pd.concat([new_dataframe, group])

    return all_stats, new_dataframe


def open_original_to_df(file, to_numeric=False, path=None):
    """
    Opens the original json file and converts it to a pandas dataframe

    Parameters:
        file (str): The name of the file to open
        to_numeric (bool): If True, converts the label to a number instead of a string (required for IQR method)
        path (str): The path to the file. If None, uses the input_directory flag
    Returns:
        pandas.DataFrame: The converted data with columns 'x', 'y', 'z', and 'label'

    """

    # read the json file
    with open(os.path.join(path, file)) as json_file:
        data = json.load(json_file)

        # convert the data to a pandas dataframe
        dataframe = pd.DataFrame()

        log.info("Converting data to dataframe")
        for label in data["coords_3d"].keys():
            if label == "com":
                continue
            dataframe = setup_utils.to_dataframe(
                dataframe, np.asanyarray(data["coords_3d"][label]["xyz"]), label
            )

        # remove nontype values from dataframe
        dataframe = dataframe[dataframe["x"] != "NoneType"]
        dataframe = dataframe[dataframe["y"] != "NoneType"]
        dataframe = dataframe[dataframe["z"] != "NoneType"]

        if to_numeric:
            # convert the label to a number
            mapping = {val: i for i, val in enumerate(dataframe["label"].unique())}

            # encode the column to an integer
            dataframe["label"] = dataframe["label"].map(mapping)

            # Convert the DataFrame to numeric dtype
            df_numeric = dataframe.apply(pd.to_numeric, errors="coerce")

            # Check if all columns are numeric now
            if df_numeric.select_dtypes(include="number").columns.size == 0:
                raise ValueError("DataFrame has no numeric columns")
            
            mapping = {v: k for k, v in mapping.items()}

            return df_numeric, mapping
        
        

        return dataframe, None


def save_to_tensor(df, filename, type, path=None):
    """
    Saves the data to a tensor file

    parameters:
        df: the dataframe to save
        folder: the folder to save the file to
        filename: the name of the file to save

    returns:
        None
    """
    if path is None:
        path = flags.FLAGS.output_directory

    df = df.rename_axis("MyIdx").sort_values(
        by=["MyIdx", "label"], ascending=[True, True]
    )

    # drop the label values
    df_tmp = df.drop(columns=["label"])
    
    # combine the columns into list
    df_tmp["combined"] = df_tmp.values.tolist()

    # join the rows with the same index
    df_tmp = df_tmp.groupby("MyIdx")["combined"].apply(list).reset_index()

    # # convert the list of coordinates into a torch tensor
    model_tensor = nn.tensor(df_tmp["combined"].values.tolist())

    if type == "hip_centered":
        folder = "hip_centered"

        # reshape the tensor so that the shape is (number of frames, 13 * 4) (13 joints, 3 coordinates + distance to hip)
        model_tensor = model_tensor.view(
            model_tensor.shape[0], model_tensor.shape[1] * model_tensor.shape[2]
        )
    else:
        folder = "coordinates"
        # reshape the tensor so that the shape is (number of frames, 13, 3) (13 joints, 3 coordinates)
        model_tensor = model_tensor.view(-1, 13, 3)

    # save the tensor
    # join the output directory with the folder
    folder = os.path.join(path, folder)

    log.info(f"Saving tensor to {folder}/{filename}.pt")

    if not os.path.exists(folder):
        log.info(f"Creating folder: {folder}")
        os.makedirs(folder)

    nn.save(model_tensor, f"{folder}/{filename}.pt")

    return


def save_to_pickle(df, filename, type, path=None):
    """ "
    Save the dataframe to pickle format.

     parameters:
        df: the dataframe to save
        folder: the folder to save the file to
        filename: the name of the file to save

    returns:
        None
    """
    if path is None:
        path = flags.FLAGS.output_directory

    df = df.rename_axis("MyIdx").sort_values(
        by=["MyIdx", "label"], ascending=[True, True]
    )

    log.info(f"Saving pickle to {path}")

    if not os.path.exists(path):
        log.info(f"Creating folder: {path}")
        os.makedirs(path)

    df.to_pickle(f"{path}/{filename}.pkl")

    return


def select_n_random_files(file_list, n):
    """
    Selects n random files from the list of files

    Args:
        file_list (list): The list of files to select from
        n (int): The number of files to select

    Returns:
        list: The list of selected files
    """

    selected_files = random.sample(file_list, n)

    return selected_files

def divide_markers(df):
    """divides the dataframe which contains all markers into a dictionary with 
    individual dataframes for each marker. 
    
    Args:
    -   df (pandas.DataFrame): the dataframe containing all markers
    
    Returns:
    -  dfs_dict (dict): a dictionary with the marker names as keys and the"""
    dfs_dict = {}

    for marker in df.label.unique():
        dfs_dict[marker] = (df.loc[(df['label']) == marker])

    return dfs_dict


def df_to_numpy(dataframe, time, label=True):
    """Converts a dataframe with the columns, xyz and label to numpy arrays
    
    Args:
    -  dataframe (pandas.DataFrame): the dataframe to convert
    - time (int): the time to select from the dataframe
    - label (bool): if true, the label column is included in the output
    
    returns:
    - X (numpy.array): the x coordinates
    - Y (numpy.array): the y coordinates
    - Z (numpy.array): the z coordinates
    - Labels (numpy.array): the labels
    """
    
    X = dataframe.loc[[time], ['x']].to_numpy()
    Y = dataframe.loc[[time], ['y']].to_numpy()
    Z = dataframe.loc[[time], ['y']].to_numpy()
    if label:
        Labels = dataframe.loc[[time], ['label']].to_numpy()
    else: 
        Labels = None
    return X, Y, Z ,Labels