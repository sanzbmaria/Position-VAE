import os
import sys
import logging as log
import pandas as pd
import numpy as np
import torch as nn

def safety_check(input_file, output_file):
    """
    Perform a safety check on the input and output folder

    :param input_file: the input folder
    :param output_file: the output folder

    :return: None

    """

    log.info('Starting safety check on the input and output folder')

    try:
        # check if the input folder exists
        if not os.path.exists(input_file):
            raise FileNotFoundError("The input folder does not exist")
    except FileNotFoundError as e:
        log.error(e)
        print(e)
        sys.exit(1)

    # check if the input file is a folder
    try:
        if not os.path.isdir(input_file):
            raise NotADirectoryError("The input file is not a folder")
    except NotADirectoryError as e:
        log.error(e)
        print(e)
        sys.exit(1)

    # check if the output file is a folder and if not create a new folder
    if not os.path.isdir(output_file):
        # print a warning
        print("The output folder does not exist, creating a new folder")
        # create the folder
        os.mkdir(output_file)

    log.info('Safety check on the input and output folder completed')



def to_dataframe(dataframe, data, label):
    """converts csv data to pandas dataframe, separating the x y z and labels"""
    log.info(f'Converting data to dataframe for label: {label}')
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

    df = group_data.copy()

    # interpolate the outliers
    # Replace outliers in the x column
    log.info(f'Interpolating outliers for label: {label} column: x')
    df = replace_outliers(df, 'x')

    # Replace outliers in the y column
    log.info(f'Interpolating outliers for label: {label} column: y')
    df = replace_outliers(df, 'y')

    # Replace outliers in the z column
    log.info(f'Interpolating outliers for label: {label} column: z')
    df = replace_outliers(df, 'z')

    # remove outlier column
    df = df.drop(['outlier_x', 'outlier_y', 'outlier_z'], axis=1)

    log.info(f'Done interpolating outliers for label: {label}')
    return df

def interpolate(dataframe):
    """interpolates the data to make it more uniform"""

    # order by index
    dataframe = dataframe.sort_index()

    # group by label
    groups = dataframe.groupby('label')

    # interpolate
    new_dataframe = pd.DataFrame()

    for name, group in groups:
        # interpolate
        group = group.drop(columns=['label'])
        group = iqr_method(group, name)

        group['label'] = name

        new_dataframe = pd.concat([new_dataframe, group])


    return new_dataframe


def save_to_tensor(df, filename, type = 'hip'):
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

    if type == 'hip':
        folder = 'hip_centered'
        # reshape the tensor so that the shape is (number of frames, 13 * 4) (13 joints, 3 coordinates + distance to hip)
        model_tensor = model_tensor.view(model_tensor.shape[0], model_tensor.shape[1]*model_tensor.shape[2])
    else:
        folder = 'coordinates'
        # reshape the tensor so that the shape is (number of frames, 13, 3) (13 joints, 3 coordinates)
        model_tensor = model_tensor.view(-1, 13, 3)

    # save the tensor
    log.info(f'Saving data {type} to tensor file in folder: {folder}')
    nn.save(model_tensor, f'../Data_52/{folder}/{filename}.pt')