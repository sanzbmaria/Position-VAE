
"""
This script processes the data from the json files and saves them as tensors after interpolating the data and centering it to the hip
"""


from absl import flags

import logging as log
import pandas as pd
import numpy as np
import torch as nn

import os
import json

from utils import safety_utils as su
from utils import data_utils as du
from utils import setup_utils

FLAGS = flags.FLAGS


def save_to_tensor(df, filename, type ):
    """
    Saves the data to a tensor file

    parameters:
        df: the dataframe to save
        folder: the folder to save the file to
        filename: the name of the file to save

    returns:
        None
    """

    df = df.rename_axis('MyIdx').sort_values(by = ['MyIdx', 'label'], ascending = [True, True])

    # drop the label values
    df_tmp = df.drop(columns=['label'])

    # combine the columns into list
    df_tmp['combined']= df_tmp.values.tolist()

    # join the rows with the same index
    df_tmp = df_tmp.groupby('MyIdx')['combined'].apply(list).reset_index()

    # # convert the list of coordinates into a torch tensor
    model_tensor = nn.tensor(df_tmp['combined'].values.tolist())

    if type == 'hip_centered':
        folder = 'hip_centered'

        # reshape the tensor so that the shape is (number of frames, 13 * 4) (13 joints, 3 coordinates + distance to hip)
        model_tensor = model_tensor.view(model_tensor.shape[0], model_tensor.shape[1]*model_tensor.shape[2])
    else:
        folder = 'coordinates'
        # reshape the tensor so that the shape is (number of frames, 13, 3) (13 joints, 3 coordinates)
        model_tensor = model_tensor.view(-1, 13, 3)

    # save the tensor
    # join the output directory with the folder
    folder = os.path.join(flags.FLAGS.output_directory, folder)

    log.info(f'Saving tensor to {folder}/{filename}.pt')

    nn.save(model_tensor, f'{folder}/{filename}.pt')


def center_hip(df):
    """
    Centers the others joints around the hip and calculates the distance between the hip and the rest of the joints

    parameters:
        df: the dataframe to center

    returns:
        df: the dataframe with the hip centered around the rest of the joints

    """

    # calculate the distance between the hip and the rest of the body parts
    groups = df.groupby('label')
    hip = groups.get_group('hip')
    hip = hip.drop(columns=['label'])

    temp_df = pd.DataFrame()
    for name, group in groups:
        if name != 'hip':
            group = group.drop(columns=['label'])
            group['distance'] =  np.sqrt(group['x']**2 + group['y']**2 + group['z']**2)
            group['label'] = name
            temp_df = pd.concat([temp_df, group], axis=0)

    # calculate the distance between the hip and the hip
    hip['distance'] =  np.sqrt(hip['x']**2 + hip['y']**2 + hip['z']**2)
    hip['label'] = 'hip'
    temp_df = pd.concat([temp_df, hip], axis=0)

    return temp_df


def main():
    # print all the flags

    log.info('Loading data to process')

    files = os.listdir(FLAGS.input_directory)

    # remove the .json from the file name
    file_name = [file[:-5] for file in files]

    # interpolate the data, center to hip and get the distance between the hip and the rest of the body parts
    for file in files:
        log.info(f'Processing file: {file}')

        df_numeric, mapping = du.open_original_to_df(file, to_numeric=True)

        log.info('Interpolating data')
         # crop values that are outside of the range (-5, 5) and replace them with -5 or 5
        df_numeric['x'] = df_numeric['x'].clip(lower=-5, upper=5)
        df_numeric['y'] = df_numeric['y'].clip(lower=-5, upper=5)
        df_numeric['z'] = df_numeric['z'].clip(lower=-5, upper=5)

        # interpolate the data
        df = du.interpolate(df_numeric)

        # check if dataframe is same as df
        if not df.equals(df_numeric):
            log.info('Dataframe has been interpolated')

        # decode the label column back to the original value
        df['label'] = df['label'].map({val: key for key, val in mapping.items()})


        # save the dataframe without centering
        save_to_tensor(df, file, type='coordinates')

        # center the hip
        log.info('Centering data to hip')
        df = center_hip(df)

        # save the dataframe with centering
        save_to_tensor(df, file, type='hip_centered')

        # check if the output directory exists if not create it
        if not os.path.exists(f'{FLAGS.output_directory}stats'):
            os.makedirs(f'{FLAGS.output_directory}stats')


    log.info('Data processing completed')

if __name__ == '__main__':
    main()