import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys 
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import time
import pyarrow.feather as feather

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a2.py')))
CurrDIR = os.path.dirname(os.path.abspath('a2.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.kmeans.kmeans import KMeansClustering
from models.pca.pca import PCA
from models.gmm.gmm import GaussianMixtureModel

from sklearn.cluster import KMeans

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/2/")

def DataLoader(DataDIR: str, datafilename: str):
    ''' 
    Loads the data from the directory. \n

    Parameters:
        DataDIR = string denoting the path to the directory containing data.
        datafilename = string denoting the name of the data file that needs to be read from DataDIR
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)

    data = pd.read_feather(data_path)
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

def VIT_Split(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function is used to split the column named 'vit' into multiple columns.

    Parameters:
        data = pandas dataframe containing the data.
    '''
    column = ['vit'+str(i) for i in range(1, 513)]
    data[column] = pd.DataFrame(data['vit'].tolist())
    data.drop('vit', axis=1, inplace=True)
    return data

########################## KMeans Clustering ##########################
# data = DataLoader(RawDataDIR, "word-embeddings.feather")
# print(data.head())

data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")

model = KMeansClustering(epochs = 10)
data_used = data.drop('words', axis=1)
model._data = data_used
err = []
err_log = []
for k in range(2, 200):
    print(k)
    model.setK(k)
    centroids, WCSS = model.fit()
    err.append(WCSS)
    err_log.append("WCSS for k = "+ str(k) +" is " + str(WCSS))

# err_log = pd.DataFrame(err_log)
# err_log.to_csv(os.path.join(CurrDIR, "err_logs/kmeans_wcss.csv"), index=False)

err_sk = []
err_log = []
for i in range(2, 200):
    model = KMeans(n_clusters = i, random_state = 42, n_init = 1, max_iter = 10, init = 'k-means++')
    model.fit(data_used)
    err_sk.append(model.inertia_)
    err_log.append("WCSS for k = "+ str(i) +" is " + str(model.inertia_))

# err_log = pd.DataFrame(err_log)
# err_log.to_csv(os.path.join(CurrDIR, "err_logs/kmeans_wcss_sklearn.csv"), index=False)

# fig = sp.make_subplots()
# fig.add_trace(go.Scatter(x = list(range(2, 30)), y = err, mode = 'lines+markers', name = 'KMeans++'))
# fig.add_trace(go.Scatter(x = list(range(2, 30)), y = er, mode = 'lines+markers', name = 'KMeans'))
# fig.update_xaxes(title_text = "K")
# fig.update_yaxes(title_text = "WCSS")
# fig.show()

# a = 2*np.ones((2, 2))
# a = pd.DataFrame(a)
# b = 3*np.ones((2, 2))
# b[0, 0] = 0
# b[1, 1] = 1
# b[0, 1] = 2
# b[1, 0] = 3
# b = pd.DataFrame(b)
# print(b)
# print(np.mean(b.values, axis = 0))