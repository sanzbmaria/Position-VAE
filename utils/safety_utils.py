import logging as log
import os

def file_exists(filename):
    """
    Check if the file exists

    Args:
        filename (str): the file name

    Returns:
        None

    """

    # check if the file exists
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError as e:
        log.error(e)
        print(e)
        return False

    log.info('File {} exists'.format(filename))

def folder_exists(folder):
    """
    Check if the folder exists

    Args:
        folder (str): the folder name

    Returns:
        None

    """

    # check if the folder exists
    try:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"The folder {folder} does not exist")
    except FileNotFoundError as e:
        log.error(e)
        print(e)
        return False

    log.info('Folder {} exists'.format(folder))

