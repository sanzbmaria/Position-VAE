import logging as log
import os
import pickle

from enum import Flag, unique

import numpy as np
import pandas as pd
import torch as nn


from absl import flags
from tqdm import tqdm

from data_preprocessing import (
    add_landmark_location,
    calculate_distance,
    center_hip,
    check_hip_centered,
)

from pandas.io.pickle import pickle as pd_pickle
from utils import data_utils as du, setup_utils


FLAGS = flags.FLAGS


def process_file(file):
    """
        Processes a given file by performing data preprocessing, interpolation, centering, and saving the data as tensors.

        The function performs the following steps:
        1. Opens the original data file and converts it to a DataFrame.
        2. Interpolates the data to fill missing values.
        3. Adds landmark location information to the data.
        4. Decodes the label column back to its original values.
        5. Saves the data in various stages (without centering, with centering) as tensors.
        6. Centers the hip joint in the data.
        7. Checks if the hip has been centered correctly and exits if not.
        8. Saves the centered data as tensors.

        Parameters:
        - file (str): The name of the file to process.

        Returns:
        - None
    """
    log.info(f"Processing file: {file}")

    df_numeric, mapping = du.open_original_to_df(file, True, flags.FLAGS.input_directory)

    log.info("Interpolating data")
    df, df_clean = du.interpolate(df_numeric)

    df_landmark = add_landmark_location(df)
    df_clean_landmark = add_landmark_location(df_clean)

    dfs = {
        "coordinates": df,
        "coordinates_clean": df_clean,
        "landmark": df_landmark,
        "landmark_clean": df_clean_landmark,
    }


    
    for key, df in tqdm(dfs.items(), desc="Processing dataframes"):
        
        df["label"] = df["label"].map(mapping, na_action='ignore')

        save_to_tensor(df, file, type=key)

        log.info("Centering data to hip")
        centered_df = center_hip(df)
        df_centered = calculate_distance(centered_df)
        
        if not check_hip_centered(df_centered):
            log.info("Hip has not been centered")
            exit()
        
        if key  == 'coordinates': 
            name = 'hip_centered'
        elif key == 'coordinates_clean':
            name = 'hip_centered_clean'
        else: 
            name = key + 'hip_centered'

        save_to_tensor(df_centered, file, type=f"{name}")

    

def save_to_tensor(df, filename, type):
    """
    Saves the data to a tensor file

    parameters:
        df: the dataframe to save
        folder: the folder to save the file to
        filename: the name of the file to save

    returns:
        None
    """

    df = df.rename_axis("MyIdx").sort_values(
        by=["MyIdx", "label"], ascending=[True, True]
    )
    
    
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


def concatenate_tensors_from_files(input_dir: str, files: list[str]) -> nn.Tensor:
    """
    Concatenates all tensors in the given list of files in the given directory and returns the result.
    Excludes the file 'joint.pt' and prints the name of any file that contains NaN values.

    Parameters:
        input_dir (str): The directory containing the input files.
        files (List[str]): A list of filenames to read tensors from.

    Returns:
        torch.Tensor: A concatenated tensor containing the data from all valid input files.

    """
    all_tensors = []
    for file in tqdm(files):
        if file == "joint.pt":
            continue
        path = os.path.join(input_dir, file)
        data = nn.load(path)
        if nn.any(nn.isnan(data)):
            print(file)
        else:
            all_tensors.append(data)

    # concatenate tensors
    tensor = nn.cat(all_tensors, dim=0)
    return tensor



def join_files():
    """
    Load and concatenate tensors from specified subdirectories, and save the resulting tensors in
    the same subdirectories as 'joint.pt' and 'joint_scaled.pt'.

    Args:
    - None

    Returns:
    - None
    """
    
    subdirs = [f for f in os.listdir(flags.FLAGS.output_directory) if os.path.isdir(os.path.join(flags.FLAGS.output_directory, f))]
    # load files

    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        input_dir = os.path.join(FLAGS.output_directory, subdir)
        files = os.listdir(input_dir)
        tensor = concatenate_tensors_from_files(input_dir, files)
        nn.save(tensor, f"{input_dir}/joint.pt")
        scaled_tensor = (tensor.float() - nn.min(tensor).float()) / (nn.max(tensor).float() - nn.min(tensor).float())
        nn.save(scaled_tensor, f"{input_dir}/joint_scaled.pt")
        
