
"""Main entry point to run the preprocessing pipeline.

The preprocessing pipeline consists of the following steps:
    1. Load the data
    2. Visualize the data
    3. Preprocess the data (remove outliers, interpolate, center to hip, get distance between hip and rest of body)
    4. Save the data

The output folder will contain the following folders:
    1. coordinates: contains the processed data without hip centering
    2. hip_centered: contains the processed data with hip centering meaning that the hip is the origin of the coordinate system

The shape of the data is (N, 13, 4) where N is the number of frames, 13 is the number of body parts and 4 is the x, y and z coordinates + the distance to the hip.

Before running, first install the dependencies listed in ../README.md.

To run this project, use the following command (example):
python3 main.py --log_directory=logs --input_directory=../data/original/ --output_directory=../data/processed/
"""


import pandas as pd
import numpy as np
import json
import os
import torch as nn
from absl import flags
from absl import app
import logging
import importlib
import sys

import proc
import utils

FLAGS = flags.FLAGS

# Define the command-line arguments
flags.DEFINE_string('log_directory', 'logs', 'Prefix for the log directory.')
flags.DEFINE_string('input_directory', '../data/original/', 'Input file directory of data to process')
flags.DEFINE_string('output_directory', '../data/processed/', 'Output file directory of processed data')

# Parse the command-line arguments
flags.FLAGS(sys.argv)


def main():

    ##############################################
    # Create logger directory
    ##############################################

    log_dir = FLAGS.log_directory

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

    logger.info(FLAGS.log_directory)
    logger.info(FLAGS.input_directory)
    logger.info(FLAGS.output_directory)


    logger.info('Log directory: {}'.format(log_dir))

    ##############################################
    # Safety check on the input and output folder
    ##############################################

    utils.safety_check(FLAGS.input_directory,  FLAGS.output_directory)

    ##############################################
    # Load the data and perform pre-processing
    ##############################################

    proc.main()



if __name__ == '__main__':
    main()