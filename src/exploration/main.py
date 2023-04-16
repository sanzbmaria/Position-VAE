"""Main entry point to run the exploration pipeline.

The exploration pipeline is a series of steps to visualize the original and pre-processed data to get a better understanding of the data.

Example:
    python3 exploration/main.py --input_directory=data --output_directory=data/plots --log_directory=exploration/logs --log_level=DEBUG

"""


import logging as log
import sys

import plots
from absl import flags

from utils import data_utils as du
from utils import setup_utils

FLAGS = flags.FLAGS


flags.DEFINE_integer(
    "video_files", 10, "Number of random files which will be selected to plot"
)
flags.DEFINE_integer(
    "time", 100, "Number of frames to plot in gif"
)
flags.DEFINE_string(
    "log_directory", "src/exploration/logs", "Prefix for the log directory."
)
flags.DEFINE_string("log_level", "INFO", "Log level to use")
flags.DEFINE_integer(
    "n_files", 100, "Number of random files which will be selected to plot"
)

# Parse the command-line arguments
flags.FLAGS(sys.argv)

# Define the command-line arguments

flags.DEFINE_string(
    "original_data_directory", "src/data/original/", "Directory of original data"
)
flags.DEFINE_string(
    "output_directory", "src/data/plots/", "Output file directory of processed data"
)
flags.DEFINE_string(
    "stats_directory", "src/data/stats", "Directory where the stats are stored"
)


def main():
    # safety check on the input and output folder and log folder
    # all of them should end with a slash
    flags.original_data_directory = setup_utils.slash_check(
        FLAGS.original_data_directory
    )
    flags.output_directory = setup_utils.slash_check(FLAGS.output_directory)
    flags.log_directory = setup_utils.slash_check(FLAGS.log_directory)
    flags.stats_directory = setup_utils.slash_check(FLAGS.stats_directory)

    setup_utils.logger_setup()
    log.info(f"Running {__file__} with arguments: {sys.argv[1:]}")

    ##############################################
    # Safety check on the input and output folder
    ##############################################
    setup_utils.create_folder(FLAGS.output_directory)
    setup_utils.create_folder(FLAGS.log_directory)

    setup_utils.safety_check(FLAGS.original_data_directory)
    setup_utils.safety_check(FLAGS.output_directory)
    setup_utils.safety_check(FLAGS.log_directory)
    setup_utils.safety_check(FLAGS.stats_directory)

    sub_folders = ["boxplot", "outliers", "pose_img", "pose_video", "stats"]

    setup_utils.setup_subfolders(FLAGS.output_directory, sub_folders)

    ##############################################
    # Visualize the data
    ##############################################

    # plots.outliers_percent()
    # plots.outliers_boxplot()
    # plots.monkey_video()
    plots.outlier_event_plot()


if __name__ == "__main__":
    main()
