
"""Main entry point to run the exploration pipeline.

The exploration pipeline is a series of steps to visualize the original and pre-processed data to get a better understanding of the data.

Example:
    python3 exploration/main.py --input_directory=data --output_directory=data/plots --log_directory=exploration/logs --log_level=DEBUG

"""


import pandas as pd
import numpy as np
import torch as nn
from absl import flags
from absl import app
import sys
import plots
import logging as log

from utils import setup_utils

FLAGS = flags.FLAGS

# Define the command-line arguments
flags.DEFINE_string('log_directory', 'exploration/logs', 'Prefix for the log directory.')
flags.DEFINE_string('input_directory', 'data/', 'Input file directory of data to process')
flags.DEFINE_string('output_directory', '.data/plots/', 'Output file directory of processed data')
flags.DEFINE_string('log_level', 'INFO', 'Log level to use')
flags.DEFINE_integer('files_to_plot', 100, 'Number of random files which will be selected to plot')

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


    sub_folders = ['boxplot', 'outliers', 'pose_img', 'pose_video', 'stats']

    setup_utils.setup_subfolders(FLAGS.output_directory, sub_folders)

    ##############################################
    # Visualize the data
    ##############################################

    plots.main()




if __name__ == '__main__':
    main()