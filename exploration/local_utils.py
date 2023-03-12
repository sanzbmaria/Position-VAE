
import logging as log
import pandas as pd
import json

def json_to_pandas(data):
    """
    Convert a json file to a pandas dataframe

    :param json_file: the json file to convert

    :return: df (pd.DataFrame): the dataframe with the json data

    """

    log.info("Converting json file to pandas dataframe")

    # convert the json data to a pandas dataframe
    df = pd.DataFrame(data)

    # rename the columns with the value in the row 'label'
    df = df.set_axis(df.loc['label'], axis=1)

    # drop the row 'label'
    df.drop('label', inplace=True)

    log.info('Json file loaded into pandas df')

    return df