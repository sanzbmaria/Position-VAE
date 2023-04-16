import logging as log
import os

import pandas as pd
import torch as nn

from utils import setup_utils


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
    df = df.set_axis(df.loc["label"], axis=1)

    # drop the row 'label'
    df.drop("label", inplace=True)

    log.info("Json file loaded into pandas df")

    return df


def open_n_original_files(input_dir, n):
    """
    Opens n random files from the original dataset.

    Args:
    ----
        input_dir (str): path to the original dataset
        n (int): number of files to open

    Returns:
    -------
        files (list): list of n random files


    """

    setup_utils.safety_check(input_dir, exist=True, is_dir=True)

    # open the folder and randomly select n files
    files = [file for file in os.listdir(input_dir) if file.endswith(".json")]

    # if the number of files is less than n, set n to the number of files
    if len(files) < n:
        log.warning("Number of files is less than n, setting n to the number of files")
        n = len(files)

    files = nn.utils.data.random_split(files, [n, len(files) - n])[0]
    files = [file for file in files if file.endswith(".json")]

    return files
