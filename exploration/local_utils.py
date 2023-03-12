
import logging as log
import pandas as pd
import json

def load_json(json_file):
    """
    Load the json file and return a pandas dataframe

    :param json_file: the json file to load

    :return: df (pd.DataFrame): the dataframe with the json data

    """

    log.info('Loading json file')

    # load the json file
    with open(json_file, 'r') as f:
        data = json.load(f)
        log.info('Json file loaded')

    # convert the json data to a pandas dataframe
    df = pd.DataFrame(data)

    # rename the columns with the value in the row 'label'
    df = df.set_axis(df.loc['label'], axis=1)

    # drop the row 'label'
    df.drop('label', inplace=True)

    log.info('Json file loaded into pandas df')

    return df