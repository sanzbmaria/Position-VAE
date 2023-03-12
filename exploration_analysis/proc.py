from absl import flags

import logging as log
import pandas as pd
import numpy as np

import os
import json
import utils



FLAGS = flags.FLAGS


def main():
    # print all the flags

    log.info('Loading data to process')

    files = os.listdir(FLAGS.input_directory)

    # remove the .json from the file name
    file_name = [file[:-5] for file in files]

    # interpolate the data, center to hip and get the distance between the hip and the rest of the body parts
    for file in files:
        log.info(f'Processing file: {file}')

        # read the json file
        with open(FLAGS.input_directory + file) as json_file:
            data = json.load(json_file)

        # convert the data to a pandas dataframe
        dataframe = pd.DataFrame()

        for label in data['coords_3d'].keys():
            if label == 'com':
                continue
            dataframe = utils.to_dataframe(dataframe, np.asanyarray(data['coords_3d'][label]['xyz']), label)


         # remove nontype values from dataframe
        dataframe = dataframe[dataframe['x'] != 'NoneType']
        dataframe = dataframe[dataframe['y'] != 'NoneType']
        dataframe = dataframe[dataframe['z'] != 'NoneType']

        # convert the label to a number
        mapping = {val: i for i, val in enumerate(dataframe['label'].unique())}


        # encode the column to an integer
        dataframe['label'] = dataframe['label'].map(mapping)

        # Convert the DataFrame to numeric dtype
        df_numeric = dataframe.apply(pd.to_numeric, errors='coerce')

        # Check if all columns are numeric now
        if df_numeric.select_dtypes(include='number').columns.size == 0:
            raise ValueError('DataFrame has no numeric columns')

        log.info('Interpolating data')
         # crop values that are outside of the range (-5, 5) and replace them with -5 or 5
        df_numeric['x'] = df_numeric['x'].clip(lower=-5, upper=5)
        df_numeric['y'] = df_numeric['y'].clip(lower=-5, upper=5)
        df_numeric['z'] = df_numeric['z'].clip(lower=-5, upper=5)

        # interpolate the data
        df = utils.interpolate(df_numeric)

        # check if dataframe is same as df
        if not df.equals(df_numeric):
            log.info('Dataframe has been interpolated')

        # decode the label column back to the original value
        df['label'] = df['label'].map({val: key for key, val in mapping.items()})

        # save the dataframe without centering
        utils.save_to_tensor(df, file, type='coordinates')

    log.info('Data processing completed')

if __name__ == '__main__':
    main()