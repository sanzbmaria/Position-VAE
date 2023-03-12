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

    for file in files:
        logging.info(f'Processing file: {file}')

        dataframe, mapping = du.open_original_to_df(file, to_numeric=True)
        mapping = {v: k for k, v in mapping.items()}
        stats = du.get_stats(dataframe, mapping)

        # save the stats to a new json file
        logging.info(f'Saving stats to file: {file}')
        with open(flags.FLAGS.output_directory + file[:-5] + '.json', 'w') as outfile:
            json.dump(stats, outfile)

    logging.info('Done')

if __name__ == "__main__":
    main()