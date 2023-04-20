import logging
import os
import sys

import numpy as np
import pandas as pd
import torch as nn
from absl import flags

import logging as log

import ultraimport
setup_utils = ultraimport('src/utils/setup_utils.py')


def open_n_original_files(input_dir, n):
    """
    Opens n random files from the original dataset.

    Args:
        input_dir (str): path to the original dataset
        n (int): number of files to open

    Returns:
        files (list): list of n random files


    """

    setup_utils.safety_check(input_dir, exist=True, is_dir=True)

    # open the folder and randomly select n files
    files = [file for file in os.listdir(input_dir) if file.endswith(".json")]

    # if the number of files is less than n, set n to the number of files
    if len(files) < n:
        log.warning(
            "Number of files is less than n, setting n to the number of files")
        n = len(files)

    files = nn.utils.data.random_split(files, [n, len(files) - n])[0]
    files = [file for file in files if file.endswith(".json")]

    return files
