
import pandas as pd
import numpy as np
import json
import logging as log
import os

from utils import setup_utils

from absl import flags

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



def iqr_method(group_data, label, replace=True):
    """interpolates the data using the IQR method

    Parameters:
        group_data (pandas.DataFrame): The input DataFrame with three columns, 'x', 'y', 'z'
        label (str): The label of the group aka the joint
        replace (bool): If True, replaces the outliers with the previous or next value that is not an outlier.

    Returns:
        pandas.DataFrame: The modified DataFrame with outliers replaced. (if replace=True)
        stats (dict): The stats of the group_data
        """

    # check if the group_data includes label
    if 'label' in group_data.columns:
        # drop the label column
        group_data = group_data.drop(['label'], axis=1)

    Q1 = group_data.quantile(0.25)
    Q3 = group_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #get the stats of each coordinate
    x_stats = group_data['x'].describe().to_dict()
    y_stats = group_data['y'].describe().to_dict()
    z_stats = group_data['z'].describe().to_dict()


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
    outlier_y = group_data[group_data['outlier_y'] == 1]
    outlier_z = group_data[group_data['outlier_z'] == 1]


    # save the stats to a dict
    stats = {
        'percentages': {
        'outlier_x': len(outlier_x)/len(group_data),
        'outlier_y': len(outlier_y)/len(group_data),
        'outlier_z': len(outlier_z)/len(group_data),
        'label': label,
        },
        'x_stats': x_stats,
        'y_stats': y_stats,
        'z_stats': z_stats
    }

    df = group_data.copy()


    if replace:
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
    """interpolates the data to make it more uniform.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame with four columns, 'x', 'y', 'z', and 'label'.

    Returns:
            pandas.DataFrame: The modified DataFrame with outliers replaced."""

    # order by index
    dataframe = dataframe.sort_index()

    # group by label
    groups = dataframe.groupby('label')

    # interpolate
    new_dataframe = pd.DataFrame()

    for name, group in groups:
        # interpolate
        group = group.drop(columns=['label'])
        group, stats = iqr_method(group, name, replace=True)

        group['label'] = name

        new_dataframe = pd.concat([new_dataframe, group])

    return new_dataframe


def get_stats(dataframe, mapping):
    """Calls the IQR method to get the stats of the data for each label

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame with four columns, 'x', 'y', 'z', and 'label'.
        mapping (dict): The mapping of the labels to the joints {key: label, value: number}
    Returns:
            dict: The stats of the data for each label."""

    # order by index
    dataframe = dataframe.sort_index()

    # group by label
    groups = dataframe.groupby('label')


    all_stats = {}


    for name, group in groups:
        # interpolate
        group = group.drop(columns=['label'])
        group, stats = iqr_method(group, mapping[name], replace=False)

        group['label'] = name

        all_stats[mapping[name]] = stats

    return all_stats


def open_original_to_df(file, to_numeric=False):
    """
    Opens the original json file and converts it to a pandas dataframe

    Parameters:
        file (str): The name of the file to open
        to_numeric (bool): If True, converts the label to a number instead of a string (required for IQR method)

    Returns:
        pandas.DataFrame: The converted data with columns 'x', 'y', 'z', and 'label'

    """

    file_path = os.path.join(flags.FLAGS.input_directory, file)
     # read the json file
    with open(file_path) as json_file:
        data = json.load(json_file)

        # convert the data to a pandas dataframe
        dataframe = pd.DataFrame()

        log.info('Converting data to dataframe')
        for label in data['coords_3d'].keys():
            if label == 'com':
                continue
            dataframe = setup_utils.to_dataframe(dataframe, np.asanyarray(data['coords_3d'][label]['xyz']), label)


        # remove nontype values from dataframe
        dataframe = dataframe[dataframe['x'] != 'NoneType']
        dataframe = dataframe[dataframe['y'] != 'NoneType']
        dataframe = dataframe[dataframe['z'] != 'NoneType']


        if to_numeric:
            # convert the label to a number
            mapping = {val: i for i, val in enumerate(dataframe['label'].unique())}

            # encode the column to an integer
            dataframe['label'] = dataframe['label'].map(mapping)

            # Convert the DataFrame to numeric dtype
            df_numeric = dataframe.apply(pd.to_numeric, errors='coerce')

            # Check if all columns are numeric now
            if df_numeric.select_dtypes(include='number').columns.size == 0:
                raise ValueError('DataFrame has no numeric columns')


            return df_numeric, mapping

        return dataframe, None