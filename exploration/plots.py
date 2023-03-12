"""
This script is used to visualize the data from the json files in order to explore the data
"""

from absl import flags
import logging as log
import torch as nn

import pandas as pd
import local_utils
from utils import safety_utils as su
import os
import matplotlib.pyplot as plt


def outliers_plot(df, title):
     # plot the average of the outliers for each joint

    df.plot(kind='bar', figsize=(10, 8))

    # set plot max y value to 100
    plt.ylim(0, 100)

    # set the x axis label
    plt.xlabel('Coordinate')

    # set the y axis label
    plt.ylabel('Percentage of outliers')

    plt.title(title)

    # add labels of the percentage of outliers
    for index, value in enumerate(df):
        plt.text(index, value, str(round(value, 2)) + '%')

    output_file = os.path.join(flags.FLAGS.output_directory, 'outliers')
    output_file = f'{output_file}/{title}.png'

    # save the plot
    plt.savefig(output_file)

    # close the plot
    plt.close()

def outliers_percent():
    """Plots statistics of the outliers in the dataset"""

    log.info('Plotting outliers')

    input_dir = flags.FLAGS.input_directory

    # join the input directory with the stats folder
    input_dir = os.path.join(input_dir, 'processed/stats')

    # print the input directory
    log.info('Input directory: {}'.format(input_dir))

    su.folder_exists(input_dir)

    data = False
    concatenated_df = pd.DataFrame()
    # load all the json files in the folder
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            path = os.path.join(input_dir, file)
            data = local_utils.load_json(path)

            # Concatenate the dataframes vertically
            concatenated_df = pd.concat([concatenated_df, data])

    # copy the index into a new column
    concatenated_df['coordinate'] = concatenated_df.index

    # reset the index
    concatenated_df.reset_index(inplace=True, drop=True)

    grouped_df = concatenated_df.groupby('coordinate').mean()

    # multiple all the values by 100
    grouped_df = grouped_df * 100

    # for each joint plot the average of the outliers
    for joint in grouped_df.columns:
        # plot the average of the outliers for each joint
        outliers_plot(grouped_df[joint], joint)

    # Calculate the row-wise mean
    grouped_df['Mean'] = grouped_df.mean(axis=1)

    # plot the average of the outliers for each joint
    outliers_plot(grouped_df['Mean'], 'Mean')

    log.info('Outliers plotted')

def outliers_boxplot():
    # open the original json files and randomly select n files
    log. info('Plotting outliers boxplot')

    input_dir = flags.FLAGS.input_directory

    input_dir = os.path.join(input_dir, 'original')

    # choose n random files to open and plot
    n = flags.FLAGS.files_to_plot
    su.folder_exists(input_dir)

    # open the folder and randomly select n files
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.json')]

    # if the number of files is less than n, set n to the number of files
    if len(files) < n:
        log.warning('Number of files is less than n, setting n to the number of files')
        n = len(files)

    files = nn.utils.data.random_split(files, [n, len(files) - n])[0]
    files = [file for file in files if file.endswith('.json')]

    """
    todo:

    1. open the json file into dataframes
        1.1 Move the to_dataframe function from local_utils to utils
    2. join them into one big dataframe
    3. plot the boxplot for each joint
    4. plot the boxplot for the mean of the joints
    """

    print(files)



def main():
    outliers_percent()
    outliers_boxplot()


if __name__ == '__main__':
    main()

