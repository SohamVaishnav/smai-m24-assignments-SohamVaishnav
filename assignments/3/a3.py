import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys 

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff

import time

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a3.py')))
CurrDIR = os.path.dirname(os.path.abspath('a3.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.MLP.mlp import *

from sklearn.preprocessing import StandardScaler

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/3/")

def DataLoader(DataDIR: str, datafilename: str):
    ''' 
    Loads the data from the directory. \n

    Parameters:
        DataDIR = string denoting the path to the directory containing data.
        datafilename = string denoting the name of the data file that needs to be read from DataDIR
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)

    data = pd.read_csv(data_path)
    print("Data has been loaded into a dataframe successfully!")
    return data

def DataWriter(Data: pd.DataFrame, DataDIR: str, datafilename: str):
    ''' 
    Writes the data to the directory. \n
    
    Parameters:
        Data = pandas dataframe containing the data.
        DataDIR = string denoting the path to the directory containing data.
        datafilename = string denoting the name of the data file that needs to be written to DataDIR
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)

    Data.to_feather(data_path)
    print("Data has been written to "+data_path+" successfully!")
    return data_path

def DataPreprocess(data: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Preprocesses the data.

    Parameters:
        data = pandas dataframe containing the data.
    '''
    data = data.dropna()
    data = data.drop_duplicates(subset = ['Id'], keep = 'first')
    data = data.drop(columns = ['Id'])

    if ('quality' in data.columns):
         temp = data['quality']
         data = data.drop(columns = ['quality'])

    for i in data.columns:
            if (type(data[i].iloc[0]) == str or data[i].iloc[0].dtype == bool):
                continue
            means = np.mean(data[i], axis = 0)
            vars = np.var(data[i], axis = 0)
            data[i] = (data[i] - means)/np.sqrt(vars)
    
    print("Data has been preprocessed successfully!")
    data = pd.concat([data, temp], axis = 1)
    return data

################################### MLP ###################################
data = DataLoader(RawDataDIR, "WineQT.csv")
print(data.shape)
print(data.describe())

# temp = data.drop(columns = ['quality'])
# labels = [temp.columns[i] for i in range(0, 12)]
# fig = px.scatter_matrix(data, dimensions = labels, color = 'quality', labels = {label: label for label in labels})
# fig.update_traces(diagonal_visible = True, showupperhalf = False)
# fig.update_layout(height = 1700, width = 1700, title_text="Pair Plot of Wine Features by Quality")
# fig.show()


data = DataPreprocess(data)
print(data.describe())

# temp = data.drop(columns = ['quality'])
# labels = [temp.columns[i] for i in range(0, 11)]
# fig = px.scatter_matrix(data, dimensions = labels, color = 'quality', labels = {label: label for label in labels})
# fig.update_traces(diagonal_visible = True, showupperhalf = False)
# fig.update_layout(height = 1700, width = 1700, title_text="Pair Plot of Wine Features by Quality")
# fig.show()


