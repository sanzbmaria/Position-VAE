import logging
import logging as log
import os
import sys

import numpy as np
import pandas as pd
import torch as nn
from absl import flags
from utils import setup_utils


def setup_subfolders(folder_path, sub_folders):
    """
    Setup the folders for the project if the sub-folders do not exist it will create them

    Params:
        folder_path (str): The path of the folder to check
        sub_folders (list): The list of sub-folders to check

    :return: None
    """

    # check if the folder exists
    if not os.path.exists(folder_path):
        # create the folder
        os.mkdir(folder_path)
        log.info(f"Created the folder {folder_path}")

    # check if the sub-folders exists
    for sub_folder in sub_folders:
        # check if the sub-folder exists
        if not os.path.exists(os.path.join(folder_path, sub_folder)):
            # create the sub-folder
            os.mkdir(os.path.join(folder_path, sub_folder))
            log.info(f"Created the folder {sub_folder}")

        else:
            log.warn(
                f"The folder {sub_folder} already exists, the files will be overwritten"
            )


def create_folder(folder_path):
    """
    Create the folder for the project if the folder does not exist it will create it

    Params:
        folder_path (str): The path of the folder to check

    :return: None
    """
    # check if the folder exist for each sub-folder
    # divide the path in sub-folders
    sub_folders = folder_path.split(os.sep)

    current = sub_folders[0]

    for folder in sub_folders[1:]:
        # check if the folder exists if not create it
        if not os.path.exists(current):
            os.mkdir(current)
            log.info(f"Created the folder {current}")
        current = os.path.join(current, folder)

    # check if the folder exists if not create it
    if not os.path.exists(current):
        os.mkdir(current)
        log.info(f"Created the folder {current}")


def open_n_original_files(input_dir, n):
    """
    Opens n random files from the original dataset.

    Args:
        input_dir (str): path to the original dataset
        n (int): number of files to open

    Returns:
        files (list): list of n random files


    """

    setup_utils.safety_check(input_dir, exist=True, is_dir=True)

    # open the folder and randomly select n files
    files = [file for file in os.listdir(input_dir) if file.endswith(".json")]

    # if the number of files is less than n, set n to the number of files
    if len(files) < n:
        log.warning(
            "Number of files is less than n, setting n to the number of files")
        n = len(files)

    files = nn.utils.data.random_split(files, [n, len(files) - n])[0]
    files = [file for file in files if file.endswith(".json")]

    return files
