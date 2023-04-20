import json
import logging as log
import os
import random
import pickle

import numpy as np
import pandas as pd
from pandas.compat import pandas
import torch as nn
from absl import flags


# called by iqr_method()
def replace_outliers(df: pandas.DataFrame, coordinate: str="z"):
    """
    Replaces outliers in a pandas DataFrame with the previous or next value that is not an outlier.

    Parameters:
        df (pandas.DataFrame): The input DataFrame with two columns, 'value' and 'is_outlier'.
        coordinate (str): The coordinate to replace the outliers in (default: 'z', options: 'x', 'y', 'z') 

    Returns:
        df (pandas.DataFrame): The modified DataFrame with outliers replaced.
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

# called by interpolate() and get_stats()
def iqr_method(group_data: pandas.DataFrame, label: str, replace: bool=True):
    """interpolates the data using the IQR method

    Parameters:
        group_data (pandas.DataFrame): The input DataFrame with three columns, 'x', 'y', 'z'
        label (str): The label of the group aka the joint
        replace (bool): If True, replaces the outliers with the previous or next value that is not an outlier (default: True)

    Returns:
        df (pandas.DataFrame): The modified DataFrame with outliers replaced. (if replace=True)
        stats (dict): The stats of the group_data
        group_data (pandas.DataFrame): The original DataFrame without outliers replaced. (if replace=False)
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

# called by file_processing.py
def interpolate(df: pandas.DataFrame):
    """interpolates the data to make it more uniform.

    Parameters:
        df (pandas.DataFrame): The input DataFrame with four columns, 'x', 'y', 'z', and 'label'.

    Returns:
        new_dataframe (pandas.DataFrame): The modified DataFrame with outliers replaced.
        new_df_outliers (pandas.DataFrame): The DataFrame with outliers marked as 1."""

    # order by index
    df = df.sort_index()

    # group by label
    groups = df.groupby("label")

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

# Called by stats.py           
def get_stats(df: pandas.DataFrame , mapping: dict):
    """Calls the IQR method to get the stats of the data for each label

    Parameters:
        df (pandas.DataFrame): The input DataFrame with four columns, 'x', 'y', 'z', and 'label'.
        mapping (dict): The mapping of the labels to the joints {key: label, value: number}
    Returns:
            all_stats (dict): The stats of the data for each label.
            new_df (pandas.DataFrame): The df containing info of whether or not the timepoint is an outlier """

    # order by index
    df = df.sort_index()

    # group by label
    groups = df.groupby("label")

    new_dataframe = pd.DataFrame()
    all_stats = {}

    for name, group in groups:
        # interpolate
        group = group.drop(columns=["label"])
        df, stats, _= iqr_method(group, mapping[name], replace=False)

        group["label"] = name

        all_stats[mapping[name]] = stats
        new_dataframe = pd.concat([new_dataframe, group])

    return all_stats, new_dataframe

# Called by stats.py  file_processing.py and plot_utils.py
def open_original_to_df(file: str , path: str, to_numeric: bool=False):
    """
    Opens the original json file and converts it to a pandas dataframe

    Parameters:
        file (str): The name of the file to open
        to_numeric (bool): If True, converts the label to a number instead of a string (required for IQR method) (default: False)
        path (str): The path to the file. If None, uses the input_directory flag

    Returns:
        dataframe (pandas.DataFrame): The converted data with columns 'x', 'y', 'z', and 'label'
        mapping (dict): The mapping of the labels to the joints {key: label, value: number} (only if to_numeric is True)

    """
    
    if path is None:
        path = FLAGS.input_directory

    # read the json file
    with open(os.path.join(path, file)) as json_file:
        data = json.load(json_file)

        # convert the data to a pandas dataframe
        dataframe = pd.DataFrame()

        log.info("Converting data to dataframe")
        for label in data["coords_3d"].keys():
            if label == "com":
                continue
            dataframe = to_dataframe(
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


def save_to_tensor(df :pd.DataFrame , filename: str, type: str, path: str=None, save_labels: bool=True):
    """
    Converts the dataframe to a tensor and saves it to the output directory


    parameters:
        df (pandas.DataFrame):  the dataframe to save
        filename (str): the name of the file to save
        type (str): the type of tensor to save (hip_centered or coordinates)
        path (str): The path to the file. If None, uses the output_directory flag
        save_labels (bool): If true, saves the labels as a pickle file
        
    side effects:
        saves the file to the folder as a tensor file with the as type/filename.pt
        
    """

    
    df = df.rename_axis("MyIdx").sort_values(
        by=["MyIdx", "label"], ascending=[True, True]
    )
    
    if path is None:
        path = flags.FLAGS.output_directory

    
    if save_labels:
        unique_labels = df['label'].unique()   
        unique_labels_dict = {i:label for i, label in enumerate(unique_labels)}
        

        folder_labels = os.path.join(flags.FLAGS.output_directory, f"labels.pkl")
    
        if not os.path.exists(folder_labels):
            # Save the dictionary as a pickle file
            with open(folder_labels, 'wb') as f:
                pickle.dump(unique_labels_dict, f)

    # drop the label values
    df_tmp = df.drop(columns=["label"])
    
    # combine the columns into list
    df_tmp["combined"] = df_tmp.values.tolist()
    # join the rows with the same index
    df_tmp = df_tmp.groupby("MyIdx")["combined"].apply(list).reset_index()

    # # convert the list of coordinates into a torch tensor
    model_tensor = nn.tensor(df_tmp["combined"].values.tolist())

    folder = type
        
    # reshape the tensor so that the shape is (number of frames, ??? ) 
    model_tensor = model_tensor.view(
        model_tensor.shape[0], model_tensor.shape[1] * model_tensor.shape[2])

    # save the tensor
    folder = os.path.join(flags.FLAGS.output_directory, folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    log.info(f"Saving tensor to {folder}/{filename}.pt")

    nn.save(model_tensor, f"{folder}/{filename}.pt")

# called by stats.py
def save_to_pickle(df : pandas.DataFrame, filename: str , path: str=None):
    """ 
    Save the dataframe to pickle format.

    parameters:
        df (pandas.Dataframe): the dataframe to save
        filename (str): the name of the file to save
        path (str): The path to the file. If None, uses the output_directory flag

    side effects:
        Saves the dataframe to the output directory as a pickle file
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

# called by plot_utils.py
def select_n_random_files(file_list , n: int = 1):
    """
    Selects n random files from the list of files

    parameters:
        file_list (list[str]): The list of files to select from 
        n (int): The number of random files to select (default: 1)

    returns:
        selected_files (list[str]): A list of paths of the randomly selected files
    """

    selected_files = random.sample(file_list, n)

    return selected_files

# CALLED by monkey_video.py
def divide_markers(df: pandas.DataFrame):
    """Divides the dataframe which contains all markers into a dictionary with 
    individual dataframes for each marker. 
    
    parameters:
        df (pandas.DataFrame): the dataframe containing all markers
    
    returns:
        dfs_dict (dict): a dictionary with the marker names as keys and the"""
    dfs_dict = {}

    for marker in df.label.unique():
        dfs_dict[marker] = (df.loc[(df['label']) == marker])

    return dfs_dict




def to_dataframe(dataframe, data, label):
    """converts csv data to pandas dataframe, separating the x y z and labels"""

    d = {'x': data[0], 'y': data[1], 'z':data[2], 'label': label}
    df = pd.DataFrame(data=d)
    dataframe = pd.concat([df, dataframe])
    return dataframe



def json_to_pandas(json_file: json):
    """
    Convert a json file to a pandas dataframe
    parameters:
        json_file (json): the json file to convert
    returns: 
        df (pd.DataFrame): the dataframe with the json data
    """

    log.info("Converting json file to pandas dataframe")

    # convert the json data to a pandas dataframe
    df = pd.DataFrame(json_file)

    # rename the columns with the value in the row 'label'
    df = df.set_axis(df.loc["label"], axis=1)

    # drop the row 'label'
    df.drop("label", inplace=True)

    log.info("Json file loaded into pandas df")

    return df