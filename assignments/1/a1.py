import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys 
import importlib

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a1.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.knn.knn import KNN
perf_measures = importlib.import_module('performance-measures')

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/")

def Dataloader(DataDIR: str, datafilename: str, train_ratio: float, test_ratio: float):
    ''' 
    Loads the data from the directory and splits it into train, validation and test sets according to
    user defined ratios. \n
    Note: Currently this function only deals with reading .csv files.

    Arguments:
    DataDIR = string denoting the path to the directory containing data.
    datafilename = string denoting the name of the data file that needs to be read from DataDIR
    train_ratio = the part of data that needs to be set aside as train data.
    test_ratio = the part of data that needs to be set aside as test data.
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)
    assert os.path.exists(data_path), f"{data_path} path is invalid!"

    if datafilename.endswith('.csv'):
        data = pd.read_csv(data_path, index_col = 0)
    
    #printing data properties
    print("Data has been read.")
    print("Data size: (rows, cols)", data.shape)
    print("Features of the data:\n", data.columns)
    # print("Class distributions: \n", data['track_genre'])

    return data

data = Dataloader(RawDataDIR, 'spotify.csv', 0.1, 0.2)
