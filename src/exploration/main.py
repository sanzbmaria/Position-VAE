"""Main entry point to run the exploration pipeline.

This module provides the functionality to run an exploration pipeline, which consists of a series of steps to visualize the original and pre-processed data. This helps to gain a better understanding of the data by creating various plots and visualizations.

The module takes input, output, and log directory paths, as well as the log level, as command-line arguments. It performs a series of safety checks on the directories and sets up the required sub-folders. Then, it visualizes the data by generating different plots using the plots module.

Example:
    python3 exploration/main.py --input_directory=data --output_directory=data/plots --log_directory=exploration/logs --log_level=DEBUG

Attributes:
    - video_files (int): Number of random files which will be selected to make a gif 
    - time (int): Number of frames to plot in gif
    - log_directory (str): Prefix for the log directory.
    - log_level (str): Log level to use
    - n_files (int): Number of random files which will be selected to plot
    - original_data_directory (str): Directory of original data
    - output_directory (str): Output file directory of processed data
    - stats_directory (str): Directory where the stats are stored

Example usage:
>>> python3 exploration/main.py --input_directory=src/data/original --output_directory=src/data/plots --log_directory=src/exploration/logs --log_level=DEBUG

"""


import logging as log
import sys

from absl import flags


import ultraimport

plots = ultraimport('src/utils/plot_utils.py')
setup_utils = ultraimport('src/utils/setup_utils.py')

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

FLAGS = flags.FLAGS

def main():
    """Main function to run the exploration pipeline."""
    
    # safety check on the input and output folder and log folder
    # all of them should end with a slash
    FLAGS.original_data_directory = setup_utils.slash_check(
        FLAGS.original_data_directory
    )
    FLAGS.output_directory = setup_utils.slash_check(FLAGS.output_directory)
    FLAGS.log_directory = setup_utils.slash_check(FLAGS.log_directory)
    FLAGS.stats_directory = setup_utils.slash_check(FLAGS.stats_directory)

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
    
    print("Plotting outliers percent")
    plots.outliers_percent(FLAGS.stats_directory)
    print("Plotting outliers boxplot")
    plots.outliers_boxplot()
    print("Plotting outliers event plot")
    plots.monkey_video()
    print("Plotting pose video")
    plots.outlier_event_plot()


if __name__ == "__main__":
    main()
