
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


FLAGS = flags.FLAGS


def to_dataframe(dataframe, data, label):
    """converts csv data to pandas dataframe, separating the x y z and labels"""

    d = {'x': data[0], 'y': data[1], 'z':data[2], 'label': label}
    df = pd.DataFrame(data=d)
    dataframe = pd.concat([df, dataframe])
    return dataframe



def replace_outliers(df, coordinate='z'):
    """
    Replaces outliers in a pandas DataFrame with the previous or next value that is not an outlier.

    Parameters:
        df (pandas.DataFrame): The input DataFrame with two columns, 'value' and 'is_outlier'.

    Returns:
        pandas.DataFrame: The modified DataFrame with outliers replaced.
    """
    # Create a new column called 'value2' that has the same values as 'value'
    df[f'{coordinate}_new'] = df[f'{coordinate}']

    # if 'outlier_{coordinate}' == 1 replace with NaN
    df.loc[df[f'outlier_{coordinate}'] == 1, f'{coordinate}_new'] = np.nan

    # Forward fill NaNs with the next non-outlier value
    df[f'{coordinate}_new'] = df[f'{coordinate}_new'].fillna(method='ffill')

    # Backward fill NaNs with the previous non-outlier value
    df[f'{coordinate}_new'] = df[f'{coordinate}_new'].fillna(method='bfill')

    # Drop the f'{coordinate}' column and rename 'value2' to 'value'
    df = df.drop(f'{coordinate}', axis=1).rename(columns={f'{coordinate}_new': f'{coordinate}'})
    return df


def iqr_method(group_data, label):
    """interpolates the data using the IQR method"""

    Q1 = group_data.quantile(0.25)
    Q3 = group_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # create a column and mark the outliers
    group_data['outlier_x'] = 0
    group_data.loc[group_data['x'] < lower_bound['x'], 'outlier_x'] = 1
    group_data.loc[group_data['x'] > upper_bound['x'], 'outlier_x'] = 1

    group_data['outlier_y'] = 0
    group_data.loc[group_data['y'] < lower_bound['y'], 'outlier_y'] = 1
    group_data.loc[group_data['y'] > upper_bound['y'], 'outlier_y'] = 1

    group_data['outlier_z'] = 0
    group_data.loc[group_data['z'] < lower_bound['z'], 'outlier_z'] = 1
    group_data.loc[group_data['z'] > upper_bound['z'], 'outlier_z'] = 1

    # send the outliers to stat
    outlier_x = group_data[group_data['outlier_x'] == 1]
    nonoutlier_x = group_data[group_data['outlier_x'] == 0]

    outlier_y = group_data[group_data['outlier_y'] == 1]
    nonoutlier_y = group_data[group_data['outlier_y'] == 0]

    outlier_z = group_data[group_data['outlier_z'] == 1]
    nonoutlier_z = group_data[group_data['outlier_z'] == 0]

    log.info(f'Percentage of outliers for label: {label} column: x: {len(outlier_x)/len(group_data)}')
    log.info(f'Percentage of outliers for label: {label} column: y: {len(outlier_y)/len(group_data)}')
    log.info(f'Percentage of outliers for label: {label} column: z: {len(outlier_z)/len(group_data)}')

    # save the stats to a dict
    stats = {
        'outlier_x': len(outlier_x)/len(group_data),
        'outlier_y': len(outlier_y)/len(group_data),
        'outlier_z': len(outlier_z)/len(group_data),
    }

    df = group_data.copy()

    # interpolate the outliers
    # Replace outliers in the x column
    df = replace_outliers(df, 'x')

    # Replace outliers in the y column
    df = replace_outliers(df, 'y')

    # Replace outliers in the z column
    df = replace_outliers(df, 'z')

    # remove outlier column
    df = df.drop(['outlier_x', 'outlier_y', 'outlier_z'], axis=1)

    log.info(f'Done interpolating outliers for label: {label}')
    return df , stats

def interpolate(dataframe):
    """interpolates the data to make it more uniform"""

    # order by index
    dataframe = dataframe.sort_index()

    # group by label
    groups = dataframe.groupby('label')

    # interpolate
    new_dataframe = pd.DataFrame()

    all_stats = {}

    for name, group in groups:
        # interpolate
        group = group.drop(columns=['label'])
        group, stats = iqr_method(group, name)

        group['label'] = name

        new_dataframe = pd.concat([new_dataframe, group])

        all_stats[name] = stats


    return new_dataframe, all_stats


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

        # read the json file
        with open(FLAGS.input_directory + file) as json_file:
            data = json.load(json_file)

        # convert the data to a pandas dataframe
        dataframe = pd.DataFrame()

        log.info('Converting data to dataframe')
        for label in data['coords_3d'].keys():
            if label == 'com':
                continue
            dataframe = to_dataframe(dataframe, np.asanyarray(data['coords_3d'][label]['xyz']), label)


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
        df, stats = interpolate(df_numeric)

        # add the label column to the respective mapping in the dict
        for key, val in mapping.items():
            stats[val]['label'] = key



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


        # save the stats file to a json file
        with open(f'{FLAGS.output_directory}stats/{file}', 'w') as f:
            json.dump(stats, f)
            f.close()

    log.info('Data processing completed')

if __name__ == '__main__':
    main()