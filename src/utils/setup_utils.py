import logging
import logging as log
import os
import sys

import numpy as np
import pandas as pd
import torch as nn
from absl import flags


def logger_setup(log_dir : str , log_level : str ):
    """
    Setup the logger for the project and creates a log file in the log directory.
    
    Params:
        log_dir (str): The path of the folder to check if None it will use the log_directory from the flags
        log_level (str): The log level to use if None it will use the log_level from the flags

    Returns:
        logger (logging): The logger object

    """

    ##############################################
    # Create logger directory
    ##############################################

    if log_dir is None:
        log_dir = flags.FLAGS.log_directory

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # If log_dir is not empty, create a new enumerated sub-directory in it for
    # logger.
    list_log_dir = os.listdir(log_dir)

    if len(list_log_dir) != 0:  # For safety, explicitly use len instead of bool
        existing_log_subdirs = [
            int(filename) for filename in list_log_dir if filename.isdigit()
        ]
        if not existing_log_subdirs:
            existing_log_subdirs = [-1]
        new_log_subdir = str(max(existing_log_subdirs) + 1)
        log_dir = os.path.join(log_dir, new_log_subdir)
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(log_dir, "0")
        os.mkdir(log_dir)

    ##############################################
    # Load config
    ##############################################

    LOG_FILENAME = r"log_file.out"
    # join the log directory with the log file name

    LOG_FILENAME = os.path.join(log_dir, LOG_FILENAME)

    logger = logging
    
    if log_level is None:
        log_level = flags.FLAGS.log_level

    logging.basicConfig(
        filename=LOG_FILENAME,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s",
        level=log_level,
        force=True,
    )

    # logger.basicConfig(filename=LOG_FILENAME, format='%(asctime)s - %(message)s', level=flags.FLAGS.log_level)

    logger.info("Log directory: {}".format(log_dir))
    return logger


def safety_check(folder_path : str , exist=True, is_dir=True):
    """
    Perform a safety check on the input and output folder

    Params:
        folder_path (str): The path of the folder to check 
        exist (bool): If True, check if the folder exists (default: True)
        is_dir (bool): If True, check if the folder is a directory, else check if it is a file (default: True)
        
    Side effects:
        If the folder does not exist it will log the error and exit the program

    """

    log.info(f"Starting safety check on {folder_path}")

    if exist:
        # is it a folder?
        if os.path.isdir(folder_path):
            try:
                # check if the folder exists
                if not os.path.exists(folder_path):
                    raise FileNotFoundError(f"The input folder {folder_path} does not exist")
            except FileNotFoundError as e:
                log.error(e)
                print(e)
                exit(1)
        else:
            try:
                # check if the file exists
                if not os.path.exists(folder_path):
                    raise FileNotFoundError(f"The file {folder_path} does not exist")
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





def slash_check(path: str):
    """
    Checks that the path does not end with a slash, if not it adds it

    Args:
        path (str): The path to check

    Returns:
        path (str): The path with a slash at the end
    """

    log.info(f"Checking if the path {path} ends with a slash")

    if path[-1] != "/":
        path += "/"
        log.info(f"Path {path} does not end with a slash, adding it")
        return path
    else:
        log.info(f"Path {path} ends with a slash")
        return path
