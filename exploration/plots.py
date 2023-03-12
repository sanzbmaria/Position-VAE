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
import numpy as np
import json
import random
from utils import setup_utils

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

    # check if the stats folder exists
    if not os.path.exists(os.path.join(input_dir, 'plots/stats')):
        log.error('The stats folder does not exist. Run the stats.py script first')
        exit()


    # join the input directory with the stats folder
    input_dir = os.path.join(input_dir, 'plots/stats')

    # print the input directory
    log.info('Input directory: {}'.format(input_dir))

    su.folder_exists(input_dir)

    data = False
    concatenated_df = pd.DataFrame()
    # load all the json files in the folder
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            path = os.path.join(input_dir, file)
             # load the json file
            with open(path, 'r') as f:
                data = json.load(f)
                log.info('Json file loaded')

            data = {key: value['percentages'] for key, value in data.items()}

            data = local_utils.json_to_pandas(data)

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
    files = [file for file in os.listdir(input_dir) if file.endswith('.json')]

    # if the number of files is less than n, set n to the number of files
    if len(files) < n:
        log.warning('Number of files is less than n, setting n to the number of files')
        n = len(files)

    files = nn.utils.data.random_split(files, [n, len(files) - n])[0]
    files = [file for file in files if file.endswith('.json')]


    all_files_df = pd.DataFrame()
    for file in files:
        log.info('Opening file: {}'.format(file))
        # read the json file
        with open(f'{flags.FLAGS.input_directory}/original/{file}') as json_file:
            data = json.load(json_file)

        # convert the data to a pandas dataframe
        dataframe = pd.DataFrame()

        log.info('Converting data to dataframe')
        for label in data['coords_3d'].keys():
            if label == 'com':
                continue
            dataframe = setup_utils.to_dataframe(dataframe, np.asanyarray(data['coords_3d'][label]['xyz']), label)

        all_files_df = pd.concat([all_files_df, dataframe])

    log.info('Finished opening files')

    # convert the columns xyz to float type
    all_files_df['x'] = all_files_df['x'].astype(float)
    all_files_df['y'] = all_files_df['y'].astype(float)
    all_files_df['z'] = all_files_df['z'].astype(float)

    # group the dataframe by the joint and drop the label column
    groups = all_files_df.groupby('label', group_keys=True).apply(lambda x: x.drop('label', axis=1))

    log.info('Plotting boxplot')

    # for each group plot the boxplot
    groups.boxplot(rot=45, fontsize=12, figsize=(8,10))

    # set the x axis label
    plt.xlabel('Coordinate')

    # set the y axis label
    plt.ylabel('Value')

    # save
    output_file = os.path.join(flags.FLAGS.output_directory, 'boxplot')

    # create the folder if it doesn't exist
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    output_file = f'{output_file}/boxplot.png'

    log.info(f'Saving boxplot to: {output_file}')

    plt.savefig(output_file)

def monkey_video():
    log.info('Plotting monkey video')

def main():
    outliers_percent()
    outliers_boxplot()
    monkey_video()


if __name__ == '__main__':
    main()

