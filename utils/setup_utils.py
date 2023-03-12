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

    logger.basicConfig(filename=LOG_FILENAME, format='%(asctime)s - %(message)s', level=logging.DEBUG)

    logger.info(flags.FLAGS.log_directory)
    logger.info(flags.FLAGS.input_directory)
    logger.info(flags.FLAGS.output_directory)


    logger.info('Log directory: {}'.format(log_dir))
    return logger

def safety_check():
    """
    Perform a safety check on the input and output folder

    :param input_file: the input folder
    :param output_file: the output folder

    :return: None

    """

    input_file = flags.FLAGS.input_directory
    output_file = flags.FLAGS.output_directory

    log.info('Starting safety check on the input and output folder')


    try:
        # check if the input folder exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"The input folder {flags.FLAGS.input_directory} does not exist")
    except FileNotFoundError as e:
        log.error(e)
        print(e)
        sys.exit(1)

    # check if the input file is a folder
    try:
        if not os.path.isdir(input_file):
            raise NotADirectoryError("The input file is not a folder")
    except NotADirectoryError as e:
        log.error(e)
        print(e)
        sys.exit(1)

    # check if the output file is a folder and if not create a new folder
    if not os.path.isdir(output_file):
        # print a warning
        print("The output folder does not exist, creating a new folder")
        # create the folder
        os.mkdir(output_file)

    # check if the output folder contains the 'coordinates' and 'hip_center' folders
    if not os.path.exists(os.path.join(output_file, 'coordinates')):
        os.mkdir(os.path.join(output_file, 'coordinates'))
    if not os.path.exists(os.path.join(output_file, 'hip_centered')):
        os.mkdir(os.path.join(output_file, 'hip_centered'))

    # check if the coordinates folder is empty
    if os.listdir(os.path.join(output_file, 'coordinates')):
        log.warning("The coordinates folder is not empty, the files will be overwritten")
    # check if the hip_center folder is empty
    if os.listdir(os.path.join(output_file, 'hip_centered')):
        log.warning("The hip_centered folder is not empty, the files will be overwritten")


    log.info('Safety check on the input and output folder completed')


def to_dataframe(dataframe, data, label):
    """converts csv data to pandas dataframe, separating the x y z and labels"""

    d = {'x': data[0], 'y': data[1], 'z':data[2], 'label': label}
    df = pd.DataFrame(data=d)
    dataframe = pd.concat([df, dataframe])
    return dataframe