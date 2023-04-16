"""
This module contains functions for calculating statistics on the data.
It is used to calculate the percentage of outliers in the data and to calculate the mean and standard deviation for each file of the data.

"""

import sys

sys.path.append("src/utils")


import json
import logging
import os
import sys

import data_utils as du
import setup_utils
from absl import flags
from tqdm import tqdm

FLAGS = flags.FLAGS

# Define the command-line arguments
flags.DEFINE_string(
    "log_directory", "src/exploration/logs", "Prefix for the log directory."
)
flags.DEFINE_string("log_level", "INFO", "Log level to use")

flags.FLAGS(sys.argv)

flags.DEFINE_string(
    "input_directory",
    "src/data/original/json/",
    "Input file directory of data to process",
)
flags.DEFINE_string(
    "output_directory", "src/data/", "Output file directory of processed data"
)


def main():
    # safety check on the input and output folder and log folder
    # all of them should end with a slash
    flags.input_directory = setup_utils.slash_check(FLAGS.input_directory)
    flags.output_directory = setup_utils.slash_check(FLAGS.output_directory)
    flags.log_directory = setup_utils.slash_check(FLAGS.log_directory)

    setup_utils.logger_setup()

    logging.info(f"Running {__file__} with arguments: {sys.argv[1:]}")

    ##############################################
    # Safety check on the input and output folder
    ##############################################
    setup_utils.create_folder(FLAGS.output_directory)
    setup_utils.create_folder(FLAGS.log_directory)

    setup_utils.safety_check(FLAGS.input_directory)
    setup_utils.safety_check(FLAGS.output_directory)
    setup_utils.safety_check(FLAGS.log_directory)

    logging.info("Loading data to process")

    files = os.listdir(flags.FLAGS.input_directory)

    is_folder = setup_utils.safety_check(
        os.path.join(flags.FLAGS.input_directory, files[0]), exist=True, is_dir=False
    )

    if is_folder:
        logging.error(
            f"The input directory {flags.FLAGS.input_directory} is a folder. Please specify the path to the files"
        )
        exit()

    for file in tqdm(files, desc='Processing files'):
        logging.info(f"Processing file: {file}")

        dataframe, mapping = du.open_original_to_df(
            file, True, flags.FLAGS.input_directory
        )
        stats, stats_df = du.get_stats(dataframe, mapping)

        output_dir_stats = os.path.join(flags.FLAGS.output_directory, "stats")
    

        # save the stats to a new json file
        
        stats_overview_path = os.path.join(output_dir_stats, "stats_overview")
        stats_df_path = os.path.join(output_dir_stats, "stats_df")
        
        # if the folder does not exist, create it
        if not os.path.exists(stats_overview_path):
            os.makedirs(stats_overview_path)
        if not os.path.exists(stats_df_path):
            os.makedirs(stats_df_path)
        
        stats_overview_path = os.path.join(stats_overview_path, file[:-5])
        stats_df_path = os.path.join(stats_df_path, file[:-5])


        logging.info(f"Saving stats to overview to: {stats_overview_path}")
        with open(stats_overview_path + ".json", "w") as outfile:
            json.dump(stats, outfile)
            
        # decode the label column back to the original value
        dataframe["label"] = dataframe["label"].map(mapping)
        stats_df["label"] = stats_df["label"].map(mapping)

        # saving the dataframe to tensor
        logging.info("Converting to tensor, and saving it")

        # remove the last folder from the input directory
        # and add 'tensor' to the end

        tensor_path = "/".join(flags.input_directory.split("/")[:-2]) + "/tensor/"
        pickle_path = "/".join(flags.input_directory.split("/")[:-2]) + "/pickle/"

        du.save_to_tensor(dataframe, file, type="coordinates", path=tensor_path)
        du.save_to_pickle(dataframe, file, type="dataframe", path=pickle_path)
        
        logging.info(f"Saving stats to dataframe to: {stats_df_path}")
        # save the pandas pf to pickle
        stats_df.to_pickle(stats_df_path + ".pkl")

    logging.info("Done")


if __name__ == "__main__":
    main()
