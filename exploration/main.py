
"""Main entry point to run the exploration pipeline.

The exploration pipeline is a series of steps to visualize the original and pre-processed data to get a better understanding of the data.

Example:
    $ python3 exploration/main.py --input_directory=data --output_directory=data/plots --log_directory=exploration/logs --log_level=INFO

"""


import pandas as pd
import numpy as np
import torch as nn
from absl import flags
from absl import app
import sys
import plots

from utils import setup_utils

FLAGS = flags.FLAGS

# Define the command-line arguments
flags.DEFINE_string('log_directory', 'logs', 'Prefix for the log directory.')
flags.DEFINE_string('input_directory', '../data/original/', 'Input file directory of data to process')
flags.DEFINE_string('output_directory', '../data/processed/', 'Output file directory of processed data')
flags.DEFINE_string('log_level', 'INFO', 'Log level to use')
flags.DEFINE_integer('files_to_plot', 100, 'Number of random files which will be selected to plot')

# Parse the command-line arguments
flags.FLAGS(sys.argv)


def main():

    setup_utils.logger_setup()

    ##############################################
    # Safety check on the input and output folder
    ##############################################

    setup_utils.safety_check()

    ##############################################
    # Visualize the data
    ##############################################

    plots.main()




if __name__ == '__main__':
    main()