
"""Main entry point to run the preprocessing pipeline.

The preprocessing pipeline consists of the following steps:
    1. Load the data
    2. Visualize the data
    3. Preprocess the data (remove outliers, interpolate, center to hip, get distance between hip and rest of body)
    4. Save the data

The output folder will contain the following folders:
    1. coordinates: contains the processed data without hip centering
    2. hip_centered: contains the processed data with hip centering meaning that the hip is the origin of the coordinate system
    4. stats: contains the statistics of the data (mean, std, min, max, percentage_of_outliers)
The shape of the data is (N, 13, 4) where N is the number of frames, 13 is the number of body parts and 4 is the x, y and z coordinates + the distance to the hip.

Before running, first install the dependencies listed in ../README.md.

To run this project, use the following command (example):
python3 preprocessing/main.py --log_directory=preprocessing/logs --input_directory=data/original/ --output_directory=data/processed/
"""


import pandas as pd
import numpy as np
import json
import os
import torch as nn
from absl import flags
from absl import app
import logging as log
import importlib
import sys

import proc
from utils import setup_utils

FLAGS = flags.FLAGS

# Define the command-line arguments
flags.DEFINE_string('log_directory', 'preprocessing/logs', 'Prefix for the log directory.')
flags.DEFINE_string('input_directory', 'data/original/', 'Input file directory of data to process')
flags.DEFINE_string('output_directory', 'data/processed/', 'Output file directory of processed data')

# Parse the command-line arguments
flags.FLAGS(sys.argv)


def main():

    setup_utils.logger_setup()
    log.info(f'Running {__file__} with arguments: {sys.argv[1:]}')

    ##############################################
    # Safety check on the input and output folder
    ##############################################

    setup_utils.create_folder(FLAGS.output_directory)
    setup_utils.create_folder(FLAGS.log_directory)


    setup_utils.safety_check(FLAGS.input_directory)
    setup_utils.safety_check(FLAGS.output_directory)
    setup_utils.safety_check(FLAGS.log_directory)

    sub_folders = ['coordinates', 'hip_centered']

    setup_utils.setup_subfolders(FLAGS.output_directory, sub_folders)


    ##############################################
    # Load the data and perform pre-processing
    ##############################################

    proc.main()



if __name__ == '__main__':
    main()