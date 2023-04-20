"""
This module contains functions for processing, preprocessing, and saving human joint data in the form of tensors.
"""

import logging as log
import os

import torch as nn

from absl import flags
from tqdm import tqdm

from data_processing import add_landmark_location, calculate_distance, center_hip, check_hip_centered, remove_rows, check_index_row_count

from ultraimport import ultraimport

du = ultraimport('src/utils/data_utils.py')


FLAGS = flags.FLAGS

def process_file(file:  str):
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
            file (str): The name of the file to process.

    """
    log.info(f"Processing file: {file}")

    df_numeric, mapping = du.open_original_to_df(file = file, path =  flags.FLAGS.input_directory, to_numeric= True)

    log.info("Interpolating data")
    
    
    df_numeric = remove_rows(df = df_numeric, n = 4, type='g')
    df_numeric = remove_rows(df = df_numeric, n = -4, type = 's')
    
    check_index_row_count(df = df_numeric,  expected_count = 13)

    df, df_clean = du.interpolate(df_numeric)

    df_landmark = add_landmark_location(df)
    df_clean_landmark = add_landmark_location(df_clean)

    dfs = {
        # "coordinates": df,
        "coordinates_clean": df_clean,
        # "landmark": df_landmark,
        "landmark_clean": df_clean_landmark,
    }

    
    for key, df in tqdm(dfs.items(), desc="Processing dataframes"):
        
        df["label"] = df["label"].map(mapping, na_action='ignore')

        du.save_to_tensor(df = df, filename = file, type=key)
        
        df = calculate_distance(df=df)

        log.info("Centering data to hip")
        centered_df = center_hip(df)
        
        
        if not check_hip_centered(centered_df):
            log.info("Hip has not been centered")
            exit()
        
        if key  == 'coordinates': 
            name = 'hip_centered'
        elif key == 'coordinates_clean':
            name = 'hip_centered_clean'
        else: 
            name = key + '_hip_centered'

        if key == list(dfs.keys())[0]:
            du.save_to_tensor(centered_df, file, type=name, save_labels=True)
        du.save_to_tensor(centered_df, file, type=f"{name}", save_labels=False)

    

def concatenate_tensors_from_files(input_dir: str, files):
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
    nn.save(tensor.float() - nn.min(tensor).float()) / (nn.max(tensor).float() - nn.min(tensor).float(), f"{input_dir}/joint_scaled.pt")



def join_files():
    """
    Load and concatenate tensors from specified subdirectories, and save the resulting tensors in
    the same subdirectories as 'joint.pt' and 'joint_scaled.pt'.

    side effects:
        saves the joint tensors to the folder as a tensor file with the joint.pt and joint_scaled.pt

    """
    
    subdirs = [f for f in os.listdir(flags.FLAGS.output_directory) if os.path.isdir(os.path.join(flags.FLAGS.output_directory, f))]
    # load files

    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        input_dir = os.path.join(FLAGS.output_directory, subdir)
        files = os.listdir(input_dir)
        #tensor = concatenate_tensors_from_files(input_dir, files)
        # nn.save(tensor, f"{input_dir}/joint.pt")
        #scaled_tensor = (tensor.float() - nn.min(tensor).float()) / (nn.max(tensor).float() - nn.min(tensor).float())
        concatenate_tensors_from_files(input_dir, files)
        
    