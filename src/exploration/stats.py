"""
This module contains functions for calculating statistics on the data.
It is used to calculate the percentage of outliers in the data and to calculate the mean and standard deviation for each file of the data.

"""


from collections.abc import Mapping
import pandas as pd
import numpy as np
import json
import os

from absl import flags
import logging

import sys

from utils import data_utils as du
from utils import setup_utils

FLAGS = flags.FLAGS

# Define the command-line arguments
flags.DEFINE_string('log_directory', 'exploration/logs', 'Prefix for the log directory.')
flags.DEFINE_string('input_directory', 'data/original/', 'Input file directory of data to process')
flags.DEFINE_string('output_directory', 'data/plots/stats/', 'Output file directory of processed data')
flags.DEFINE_string('log_level', 'INFO', 'Log level to use')

# Parse the command-line arguments
flags.FLAGS(sys.argv)

def main():
    # safety check on the input and output folder and log folder
    # all of them should end with a slash
    flags.input_directory = setup_utils.slash_check(FLAGS.input_directory)
    flags.output_directory = setup_utils.slash_check(FLAGS.output_directory)
    flags.log_directory = setup_utils.slash_check(FLAGS.log_directory)

    setup_utils.logger_setup()

    logging.info(f'Running {__file__} with arguments: {sys.argv[1:]}')

    ##############################################
    # Safety check on the input and output folder
    ##############################################
    setup_utils.create_folder(FLAGS.output_directory)
    setup_utils.create_folder(FLAGS.log_directory)


    setup_utils.safety_check(FLAGS.input_directory)
    setup_utils.safety_check(FLAGS.output_directory)
    setup_utils.safety_check(FLAGS.log_directory)


    logging.info('Loading data to process')

    files = os.listdir(flags.FLAGS.input_directory)

    is_folder = setup_utils.safety_check(os.path.join(flags.FLAGS.input_directory, files[0]), exist=True, is_dir=False)

    if is_folder:
        logging.error(f'The input directory {flags.FLAGS.input_directory} is a folder. Please specify the path to the files')
        exit()


    for file in files:
        logging.info(f'Processing file: {file}')

        dataframe, mapping = du.open_original_to_df(file, to_numeric=True)
        mapping = {v: k for k, v in mapping.items()}
        stats = du.get_stats(dataframe, mapping)

        # save the stats to a new json file
        file_path = os.path.join(flags.FLAGS.output_directory,  file[:-5])

        logging.info(f'Saving stats to file: {file_path}')
        with open(file_path + '.json', 'w') as outfile:
            json.dump(stats, outfile)

    logging.info('Done')

if __name__ == "__main__":
    main()