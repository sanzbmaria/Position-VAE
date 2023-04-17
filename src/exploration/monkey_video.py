"""
This module contains functions to plot 3D scatter plots and connecting lines from a pandas DataFrame.

This module contains the following constants:
    - JOINT_COLORS (dict): A dictionary mapping joint names to their corresponding color values.
    - LIMB_COLORS (dict): A dictionary mapping limb names to their corresponding color values.
    - CONNECTIONS (list): A list of lists containing pairs of joints to connect.

This module contains the following functions:
    - connecting_lines(df, connections): A function that takes a pandas DataFrame with all markers and a list of connections, and returns a dictionary of concatenated DataFrames.
    - plot_video(df, time, file_name): A function that plots a 3D scatter plot with connecting lines between joints for each frame in a given time range and saves the animation as a GIF file.
    - plot_frame(df, con_df, i, frames): A function that plots a single frame of a 3D scatter plot with connecting lines.
"""
import os
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import pandas as pd
import imageio
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
import gif

from absl import flags
import logging as log

from utils import data_utils as du


JOINT_COLORS = {'nose': 'lightcoral',
            'head': 'coral', 
            'neck': 'sienna', 
            'RShoulder': 'plum', 
            'RHand': 'tomato', 
            'Lshoulder': 'darkslateblue', 
            'Lhand': 'salmon', 
            'hip': 'seagreen', 
            'RKnee': 'mediumspringgreen',
            'RFoot': 'springgreen', 
            'LKnee': 'mediumvioletred',
            'Lfoot': 'deeppink', 
            'tail': 'greenyellow'}

LIMB_COLORS = {'Lfoot-LKnee': 'darkviolet', 
            'RKnee-hip': 'darkviolet' , 
            'RFoot-RKnee': 'darkviolet',
            'LKnee-hip' : 'darkviolet',
            'RShoulder-neck': 'orange',
            'Lhand-Lshoulder': 'orange',
            'Lshoulder-neck': 'orange', 
            'RHand-RShoulder': 'orange',
            'hip-neck': 'lightgreen',
            'hip-tail': 'forestgreen',
            'neck-nose': 'lightblue', 
            'nose-head': 'steelblue', }

CONNECTIONS = [['Lfoot', 'LKnee'], 
                ['LKnee', 'hip'],
                ['RFoot', 'RKnee'],
                ['RKnee', 'hip'],
                ['hip', 'neck'],
                ['hip', 'tail'],
                ['neck', 'nose'],
                ['nose', 'head'],
                ['Lhand','Lshoulder'],
                ['Lshoulder', 'neck'],
                ['RHand','RShoulder'],
                ['RShoulder', 'neck' ]]

def connecting_lines(df : pandas.DataFrame, connections : list) -> dict:
    """
    Concatenates specified columns from a pandas dataframe `df` and returns a dictionary `c_dict` with keys as 'column_1-column_2' (e.g. 'A-B') and values as the concatenated dataframe for each connection.

    Parameters:
        df (pandas.DataFrame): The dataframe that contains all the joint markers and their corresponding x, y, and z coordinates.
        connections (list): A list of connections (each represented as a tuple of two column names) to be concatenated

    Returns:
        dict (dict): A dictionary where keys are in the format 'column_1-column_2' and values are concatenated dataframes.
    """
    
    c_dict = {}

    for connection in connections:
        c_dict[f'{connection[0]}-{connection[1]}'] = pd.concat([df[connection[0]], df[connection[1]]], axis=1).drop(columns=['label'])
    
    return c_dict

def plot_video(df : pd.DataFrame, time: int, file_name: str):
    """
    Generates a GIF of a 3D skeleton animation using a dataframe `df` containing x, y, and z position data
    for each joint in the monkey skeleton. The GIF covers a `time`-second time window and is saved under the 
    `file_name` provided in the output directory.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the position data for each joint in the skeleton
        time (int): The time in seconds to cover in the generated GIF
        file_name (str): The name to use for the generated GIF file

    Side Effects:
        - Creates a directory 'gifs' and 'pics' in the output directory if they don't already exist.
        - Saves a GIF file and individual PNG frames for each time point in the `time`-second window
        in the 'gifs' and 'pics' directories, respectively.

    """
    #1. divide the dataframe into a dictionary of dataframes, one for each joint
    dfs_dict = du.divide_markers(df)

    #2. create a dictionary of lines connecting the joints
    c_dict = connecting_lines(dfs_dict, CONNECTIONS)
    
    # 3. concatinate all of the dfs 
    frames = []
    for key in c_dict.keys():
        temp_b = pd.DataFrame.from_dict(c_dict[key])
        temp_b['labels'] = key
        frames.append(temp_b)
        plt.close('all')

    connecting_lines_df = pd.concat(frames).sort_index()
    
    # 4. rename the columns
    connecting_lines_df.columns = ['x1', 'y1','z1','x2', 'y2', 'z2', 'label']
    
    # 5. map the labels to a unique color value for each joint
    connecting_lines_df.loc[:, 'color']  = connecting_lines_df['label'].map(LIMB_COLORS)
        
    # map the labels to a unique color value for each joint
    df.loc[:, 'color'] = df['label'].map(JOINT_COLORS)
    
    # choose a random time point to start the gif 
    start_time = random.randint(0, df.index[-1] - time)
    end_time = start_time + time
    
    df = df[(df.index >= start_time) & (df.index < end_time)]
    
    # 6. create the folder to save the gif
    gif_path = os.path.join(flags.FLAGS.output_directory, 'gifs')
    gif_path = os.path.join(gif_path, file_name)
    
    pic_path = os.path.join(flags.FLAGS.output_directory, 'pics')
    pic_path = os.path.join(pic_path, file_name)
    
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
        
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
        
        
    log.info(f"Creating the gif for {file_name}")
    print(f"Creating the gif for {file_name}")
    log.info(f"save path: {gif_path}")
    log.info(f"image path: {pic_path}")
    
    # 7. create and save the gif
    
    with imageio.get_writer(f'{gif_path}/{start_time}_{end_time}.gif', mode='I',  loop=False) as writer:
        for i in tqdm(range(start_time, end_time)):
            fig = plot_frame(df, connecting_lines_df, i, frames)
            fig.write_image(f'{pic_path}/{i}.png')
            image = imageio.imread(f'{pic_path}/{i}.png')
            writer.append_data(image)
            matplotlib.pyplot.close()

def plot_frame(df : pandas.DataFrame = None, con_df : pandas.DataFrame = None, i : int= None, frames : int = None):
    """
    Plots a single frame of 3D scatter plot of the coordinates and connections between them.

    Args:
        df (pandas.Dataframe): A pandas DataFrame containing the coordinates for a single frame.
        con_df (pandas.Dataframe): A pandas DataFrame containing the connections between the coordinates.
        i (int): An integer indicating the index of the frame.
        frames (int): An integer indicating the total number of frames in the animation.

    Returns:
        A plotly Figure object containing the scatter plot of the coordinates and connections.
        
    Side effects:
        - Converts the coordinate points and connection points to NumPy arrays.
        - Creates a 3D scatter plot of the coordinates and connections using the Plotly library.
        - Logs progress and errors.

    """


    # convert the points to numpy 
    df = df.loc[[i]]
    con_df = con_df.loc[[i]]
    
    con_df['x'] = con_df[['x1', 'x2']].values.tolist()
    con_df['y'] = con_df[['y1', 'y2']].values.tolist()
    con_df['z'] = con_df[['z1', 'z2']].values.tolist()
    
    # create the figure
    # with a specific size
    fig = plt.figure(figsize=(10, 10))
    
    ax = plt.axes(projection='3d')
    
    # Set names for axis 
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    fig = px.scatter_3d(df, x='x', y='y', z='z',  color='color')
    fig.update_traces(marker=dict(size=4))
    
    for index, row in con_df.iterrows():
        tem = pd.DataFrame(row).T

        x = tem['x'].values[0]
        y = tem['y'].values[0]
        z = tem['z'].values[0]
        label = tem['label'].values[0]
        
        fig.add_trace(go.Scatter3d(x=x, y=y,z=z,mode='lines'))

    fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=10, range=[-5,5],),
        zaxis = dict(nticks=10, range=[-5,5],),),
        height = 800,
        width=800)
    fig.update_layout(showlegend=False)
    
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    
    
    # adjust the plot position automatically
    plt.tight_layout()
    

    return fig
