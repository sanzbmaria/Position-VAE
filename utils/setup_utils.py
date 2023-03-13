import os
import sys
import logging as log
import pandas as pd
import numpy as np
import torch as nn
import logging
from absl import flags



def logger_setup():
    """
        Setup the logger for the project and creates a log file in the log directory.

        Params:
            None
        Returns:
            logger (logging): The logger object

    """

    ##############################################
    # Create logger directory
    ##############################################

    log_dir = flags.FLAGS.log_directory

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # If log_dir is not empty, create a new enumerated sub-directory in it for
    # logger.
    list_log_dir = os.listdir(log_dir)

    if len(list_log_dir) != 0:  # For safety, explicitly use len instead of bool
        existing_log_subdirs = [
            int(filename) for filename in list_log_dir if filename.isdigit()]
        if not existing_log_subdirs:
            existing_log_subdirs = [-1]
        new_log_subdir = str(max(existing_log_subdirs) + 1)
        log_dir = os.path.join(log_dir, new_log_subdir)
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(log_dir, '0')
        os.mkdir(log_dir)

    ##############################################
    # Load config
    ##############################################

    LOG_FILENAME = r'log_file.out'
    # join the log directory with the log file name

    LOG_FILENAME = os.path.join(log_dir, LOG_FILENAME)

    logger = logging

    logger.basicConfig(filename=LOG_FILENAME, format='%(asctime)s - %(message)s', level=flags.FLAGS.log_level)

    logger.info(flags.FLAGS.log_directory)
    logger.info(flags.FLAGS.input_directory)
    logger.info(flags.FLAGS.output_directory)


    logger.info('Log directory: {}'.format(log_dir))
    return logger

def safety_check(path, exist=True, is_dir=True):
    """
    Perform a safety check on the input and output folder

    Params:
        folder_path (str): The path of the folder to check
        exist (bool): If True, check if the folder exists
        is_dir (bool): If True, check if the folder is a directory, else check if it is a file


    Returns:
        None

    """

    log.info(f'Starting safety check on {path}')

    if exist:
        try:
            # check if the folder exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"The input folder {path} does not exist")
        except FileNotFoundError as e:
            log.error(e)
            print(e)
            exit(1)

    if is_dir:
        # check if the input file is a folder
        try:
            if not os.path.isdir(path):
                raise NotADirectoryError(f"The {path} is not a folder")
        except NotADirectoryError as e:
            log.error(e)
            print(e)
            sys.exit(1)
    else:
        # check if the input file is a file
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"The {path} is not a file")
        except FileNotFoundError as e:
            log.error(e)
            print(e)
            sys.exit(1)


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
            log.warn(f"The folder {sub_folder} already exists, the files will be overwritten")


def create_folder(folder_path):
    """
    Create the folder for the project if the folder does not exist it will create it

    Params:
        folder_path (str): The path of the folder to check

    :return: None
    """

    # check if the folder exists
    if not os.path.exists(folder_path):
        # create the folder
        os.mkdir(folder_path)
        log.info(f"Created the folder {folder_path}")


def to_dataframe(dataframe, data, label):
    """converts csv data to pandas dataframe, separating the x y z and labels"""

    d = {'x': data[0], 'y': data[1], 'z':data[2], 'label': label}
    df = pd.DataFrame(data=d)
    dataframe = pd.concat([df, dataframe])
    return dataframe