"""
This script is used to visualize the data from the json files in order to explore the data
"""

import json
import logging as log
import os

import local_utils
import local_utils as lu
import matplotlib.pyplot as plt
import monkey_video as mkv
import numpy as np
import pandas as pd
from absl import flags

from utils import data_utils as du
from utils import setup_utils
from tqdm import tqdm
import seaborn as sns


def outliers_plot(df, title):
    """
    Plot the average percentage of outliers for each joint in a bar chart.

    Args:
        df (pandas.DataFrame): A DataFrame containing the average percentage of outliers for each joint.
        title (str): The title of the plot.

    Side effects:
        - Saves the plot as a PNG file in the output directory specified in the flags.
    """
    # plot the average of the outliers for each joint

    df.plot(kind="bar", figsize=(10, 8))

    # set plot max y value to 100
    plt.ylim(0, 100)

    # set the x axis label
    plt.xlabel("Coordinate")

    # set the y axis label
    plt.ylabel("Percentage of outliers")

    plt.title(title)

    # add labels of the percentage of outliers
    for index, value in enumerate(df):
        plt.text(index, value, str(round(value, 2)) + "%")

    output_file = os.path.join(flags.FLAGS.output_directory, "outliers")
    output_file = f"{output_file}/{title}.png"

    # save the plot
    plt.savefig(output_file)

    # close the plot
    plt.close()


def outliers_percent():
    """
    Loads JSON files containing outlier statistics and plots the average percentage of outliers for each joint and the overall mean.

    Raises:
        FileNotFoundError: If the stats directory or its overview subdirectory does not exist.

    Side effects:
        - Saves individual joint outliers plots and an overall mean outliers plot as PNG files in the output directory specified in the flags.
        - Logs progress and errors.

    """

    log.info("Plotting outliers")

    stats_dir = flags.stats_directory
    stats_overview_dir = os.path.join(stats_dir, 'stats_overview')

    # check if the stats folder exists
    if not os.path.exists(stats_overview_dir):
        log.error(
            f"The stats directory {stats_overview_dir} does not exist. Run the stats.py script first"
        )
        exit()

    log.info("Stats directory: {}".format(stats_overview_dir))

    data = False
    concatenated_df = pd.DataFrame()
    # load all the json files in the folder
    for file in os.listdir(stats_overview_dir):
        if file.endswith(".json"):
            path = os.path.join(stats_overview_dir, file)
            # load the json file
            with open(path, "r") as f:
                data = json.load(f)
                log.info("Json file loaded")

            data = {key: value["percentages"] for key, value in data.items()}

            data = local_utils.json_to_pandas(data)

            # Concatenate the dataframes vertically
            concatenated_df = pd.concat([concatenated_df, data])

    # copy the index into a new column
    concatenated_df["coordinate"] = concatenated_df.index

    # reset the index
    concatenated_df.reset_index(inplace=True, drop=True)

    grouped_df = concatenated_df.groupby("coordinate").mean()

    # multiple all the values by 100
    grouped_df = grouped_df * 100

    # for each joint plot the average of the outliers
    for joint in grouped_df.columns:
        # plot the average of the outliers for each joint
        outliers_plot(grouped_df[joint], joint)

    # Calculate the row-wise mean
    grouped_df["Mean"] = grouped_df.mean(axis=1)

    # plot the average of the outliers for each joint
    outliers_plot(grouped_df["Mean"], "Mean")

    log.info("Outliers plotted")


def outliers_boxplot():
    """
    Loads a specified number of original JSON files and plots a boxplot for each joint coordinate.
    
    Side effects:
        - Opens and concatenates a specified number of randomly selected original JSON files.
        - Converts the x, y, and z columns of the resulting DataFrame to float type.
        - Groups the DataFrame by joint coordinate and plots a boxplot for each group.
        - Saves the boxplot as a PNG file in the output directory specified in the flags.
        - Logs progress and errors.

    """

    # open the original json files and randomly select n files
    log.info("Plotting outliers boxplot")

    json_path = os.path.join(flags.FLAGS.original_data_directory, "json")

    files = lu.open_n_original_files(json_path, flags.FLAGS.n_files)

    all_files_df = pd.DataFrame()
    for file in files:
        dataframe, _ = du.open_original_to_df(file, False, json_path)

        all_files_df = pd.concat([all_files_df, dataframe])

    log.info("Finished opening files")

    # convert the columns xyz to float type
    all_files_df["x"] = all_files_df["x"].astype(float)
    all_files_df["y"] = all_files_df["y"].astype(float)
    all_files_df["z"] = all_files_df["z"].astype(float)

    # group the dataframe by the joint and drop the label column
    groups = all_files_df.groupby("label", group_keys=True).apply(
        lambda x: x.drop("label", axis=1)
    )

    log.info("Plotting boxplot")

    # for each group plot the boxplot
    groups.boxplot(rot=45, fontsize=12, figsize=(8, 10))

    # set the x axis label
    plt.xlabel("Coordinate")

    # set the y axis label
    plt.ylabel("Value")

    # save
    output_file = os.path.join(flags.FLAGS.output_directory, "boxplot")

    # create the folder if it doesn't exist
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    output_file = f"{output_file}/boxplot.png"

    log.info(f"Saving boxplot to: {output_file}")

    plt.savefig(output_file)


def outlier_event_plot():
    """
    Loads pickled DataFrames containing outlier statistics and plots the events on separate heatmaps for each joint coordinate.
    Raises:
        FileNotFoundError: If the stats directory or its DataFrame subdirectory does not exist.

    Side effects:
        - Loads all pickled DataFrames from the DataFrame subdirectory of the stats directory specified in the flags.
        - Concatenates the DataFrames vertically.
        - Plots the outliers events on separate heatmaps for each joint coordinate.
        - Saves the heatmaps as PNG files in the output directory specified in the flags.
        - Logs progress and errors.

    """

    # open the stats folder
    log.info("Plotting outliers in historgram")
    print('Ploting Outlier Events')

    stats_dir = flags.stats_directory
    stats_df_dir = os.path.join(stats_dir, 'stats_df')

    # check if the stats folder exists
    if not os.path.exists(stats_df_dir):
        log.error(
            f"The stats directory {stats_df_dir} does not exist. Run the stats.py script first"
        )
        exit()

    log.info(f"Stats directory: {stats_df_dir}")

    concatenated_df = pd.DataFrame()

    # load all the json files in the folder
    for file in tqdm(os.listdir(stats_df_dir), desc='Processing files'):
        if file.endswith(".pkl"):
            path = os.path.join(stats_df_dir, file)
            # load the pandas dataframe
            data = pd.read_pickle(path)

            # drop the 'x' y and 'z' columns
            data = data.drop(['x', 'y', 'z'], axis=1)

            # Concatenate the dataframes vertically
            concatenated_df = pd.concat([concatenated_df, data])

    # get a list of unique joints in the DataFrame

    # get a list of unique joints in the DataFrame
    joints = concatenated_df['label'].unique()

    # save
    output_file = os.path.join(flags.FLAGS.output_directory, "histogram")

    # create the folder if it doesn't exist
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    # iterate through each joint with tqdm
    for i, joint in enumerate(tqdm(joints, desc='Creating XYZ heatmaps')):

        fig, ax = plt.subplots(figsize=(5*len(joints), 5))

        # subset the DataFrame by joint
        subset = concatenated_df[concatenated_df['label'] == joint]

        # remove the label
        subset = subset.drop('label', axis=1)

        # get the indexes that are outliers
        x_np = subset['outlier_x'].to_numpy()
        y_np = subset['outlier_y'].to_numpy()
        z_np = subset['outlier_z'].to_numpy()

        spike_times_x = [i for i, x in enumerate(x_np) if x == 1]
        spike_times_y = [i for i, x in enumerate(y_np) if x == 1]
        spike_times_z = [i for i, x in enumerate(z_np) if x == 1]

        # plot the lines
        plt.vlines(spike_times_x, 0, 0.5, color='r')
        plt.vlines(spike_times_y, 0, 0.5, color='b')
        plt.vlines(spike_times_z, 0, 0.5, color='g')

        # set the title of the current subplot
        plt.set_title(f'{joint}')

        plt.set_ylabel('XYX')

        plt.set_xlim([0, len(x_np)])
        plt.set_xlabel('Time 30fps')

        fig.tight_layout()

        fig.savefig(f'{output_file}/{joint}_XYZ.png')


def monkey_video():
    """
    Loads pickled DataFrames containing processed video data and creates a video by joining the frames.

    Raises:
        FileNotFoundError: If the original data directory or its pickle subdirectory does not exist.

    Side effects:
        - Loads n random pickled DataFrames from the pickle subdirectory of the original data directory specified in the flags.
        - Calls the `plot_video()` function from `monkey_video` module to join the frames and create the output video.
        - Logs progress and errors.

    """

    log.info("Plotting monkey video")

    # 1. The data is already processed and saved in the data folder (by the stats.py script)
    # 2. We need to load n random data
    # 3. create a video with the frames

    dataframe_path = os.path.join(flags.original_data_directory, "pickle")

    # check if the folder is empty
    try:
        if len(os.listdir(dataframe_path)) == 0:
            log.error(
                f"The directory {dataframe_path} does not exist. Run the stats.py script first"
            )
    except FileNotFoundError:
        raise Exception(f"Directory {dataframe_path} does not exist")

    if len(os.listdir(dataframe_path)) == 0:
        log.error(
            "{dataframe_path} Folder is empty, run the stats.py script first")
        exit(0)

    # select n random files

    n = flags.FLAGS.video_files

    # open the folder and get the list of files

    file_list = os.listdir(dataframe_path)

    if len(file_list) < n:
        log.warning(
            f"Number of files is less than {n}, using {len(file_list)} files")
        n = len(file_list)

    files = du.select_n_random_files(file_list, n)

    # Open the pickle file
    for file in tqdm(files, desc='Processing files'):
        print("Opening file: {}".format(os.path.join(dataframe_path, file)))
        my_df = pd.read_pickle(os.path.join(dataframe_path, file))

        # remove the pkl and json extension from the file name
        file = file.replace(".pkl", "").replace(".json", "")

        mkv.plot_video(my_df, flags.FLAGS.time, file)

    log.info(f"Loading data for monkey video from {dataframe_path}")
    print(f"Loading data for monkey video from {dataframe_path}")

    print(files)



if __name__ == "__main__":
    main()
