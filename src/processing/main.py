
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
Or 
(N, 13, 8) where N is the number of frames, 13 is the number of body parts and 8 are the x, y and z coordinates, distance to each of the landmarks (4 barrels and 4 feeders) and distanc.e to the hip

Example:
    python3 src/preprocessing/main.py --log_directory=src/preprocessing/logs --input_directory=src/data/original/ --output_directory=src/data/processed/
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

from file_processing import join_files

from utils import setup_utils
from tqdm import tqdm

FLAGS = flags.FLAGS

# Define the command-line arguments
flags.DEFINE_string('log_directory', 'src/processing/logs', 'Prefix for the log directory.')
flags.DEFINE_string('log_level', 'INFO', 'Log level to use')

# Parse the command-line arguments
flags.FLAGS(sys.argv)

flags.DEFINE_string('input_directory', 'src/data/original/json', 'Input file directory of data to process')
flags.DEFINE_string('output_directory', 'src/data/processed/', 'Output file directory of processed data')


FLAGS = flags.FLAGS

def main():
    # safety check on the input and output folder and log folder
    # all of them should end with a slash
    FLAGS.input_directory = setup_utils.slash_check(FLAGS.input_directory)
    FLAGS.output_directory = setup_utils.slash_check(FLAGS.output_directory)
    FLAGS.log_directory = setup_utils.slash_check(FLAGS.log_directory)


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
    #############################################
    
    # print all the flags

    log.info("Loading data to process")

    files = os.listdir(FLAGS.input_directory)

    # remove the .json from the file name
    file_name = [file[:-5] for file in files]

    # # interpolate the data, center to hip and get the distance between the hip and the rest of the body parts
    # for file in tqdm(files):
    #     process_file(file)

        
    log.info("Data processing completed")
    
    ##############################################
    # Join the data and save it
    ##############################################
    
    join_files()
    
    

if __name__ == '__main__':
    main()