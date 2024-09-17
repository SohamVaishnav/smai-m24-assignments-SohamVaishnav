import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.feather as feather
from scipy.cluster.hierarchy import dendrogram, linkage

import os
import sys 

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff

import time

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a2.py')))
CurrDIR = os.path.dirname(os.path.abspath('a2.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.kmeans.kmeans import KMeansClustering
from models.pca.pca import PCA
from models.gmm.gmm import GaussianMixtureModel

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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

# data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")
# print(data.head())

# data = pd.read_csv(os.path.join(RawDataDIR, "data.csv")) #test dataset from Kaggle

# model = KMeansClustering(epochs = 10)
# data_used = data.drop('color', axis=1)
# data_used = data.drop(columns=['words'])
# model._data = data_used.copy()
# err = []
# err_log = []
# for k in range(1, 30):
#     print(k)
#     model.setK(k)
#     centroids, WCSS, _ = model.fit()
#     err.append(WCSS)
#     err_log.append("WCSS for k = "+ str(k) +" is " + str(WCSS))

# # err_log = pd.DataFrame(err_log)
# # err_log.to_csv(os.path.join(CurrDIR, "err_logs/kmeans_wcss.csv"), index=False)

# err_sk = []
# err_log = []
# for i in range(1, 30):
#     model = KMeans(n_clusters = i, random_state = 42, n_init = 1, max_iter = 10, init = 'k-means++')
#     model.fit(data_used)
#     err_sk.append(model.inertia_)
#     err_log.append("WCSS for k = "+ str(i) +" is " + str(model.inertia_))

# # err_log = pd.DataFrame(err_log)
# # err_log.to_csv(os.path.join(CurrDIR, "err_logs/kmeans_wcss_sklearn.csv"), index=False)

# fig = sp.make_subplots()
# fig.add_trace(go.Scatter(x = list(range(1, 100)), y = err, mode = 'lines+markers', name = 'KMeans_self'))
# fig.add_trace(go.Scatter(x = list(range(1, 100)), y = err_sk, mode = 'lines+markers', name = 'KMeans_Sklearn'))
# fig.update_xaxes(title_text = "K")
# fig.update_yaxes(title_text = "WCSS")
# fig.update_layout(title_text = "WCSS vs K for KMeans")
# fig.show()

# kmeans1 = 6 #got from the plot
# model.setK(kmeans1)
# centroids, WCSS, _ = model.fit()
# print("WCSS: ", WCSS)
# print(centroids)
# print("Clusters: ", model._clusters)

# fig = px.scatter(x = data_used.iloc[:, 0], y = data_used.iloc[:, 1], color = model._clusters)
# fig.add_trace(go.Scatter(x = centroids[:, 0], y = centroids[:, 1], mode = 'markers', marker = dict(size = 10)))
# fig.update_layout(title_text = "KMeans Clustering")
# fig.show()

# model = KMeans(n_clusters = kmeans1, random_state = 42, n_init = 1, max_iter = 10, init = 'k-means++')
# model.fit(data_used)
# print("WCSS: ", model.inertia_)
# print("Centroids: ", model.cluster_centers_)
# print("Labels: ", model.labels_)
# print("Iterations: ", model.n_iter_)

# fig = px.scatter(x = data_used.iloc[:, 0], y = data_used.iloc[:, 1], color = model.labels_)
# fig.add_trace(go.Scatter(x = model.cluster_centers_[:, 0], y = model.cluster_centers_[:, 1], mode = 'markers', marker = dict(color = 'black', size = 10)))
# fig.update_layout(title_text = "KMeans Clustering")
# fig.show()


########################## GMM ##########################

model = GaussianMixtureModel()
data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")
data_used = data.drop(columns=['words'])
# # data = pd.read_csv(os.path.join(RawDataDIR, "data.csv")) #test dataset from Kaggle
# # data_used = data.drop('color', axis=1)

# pca = PCA(2)
# pca.fit(data_used)
# data_used = pca.transform()

ll_self = []
for k in range(1, 10):
    print(k)
    model.fit(data_used, K = k, epochs = 4, epsilon=1e-2)
    model.getLikelihood(epsilon=1e-2)
    ll_self.append(model._likelihood)
    # print(model.getMembership(epsilon=1e-2), "\n\n")
    # print(model.getParams()[2])
    # print(model.getCluster())

ll_sk = []
for k in range(1, 10):
    model = GaussianMixture(n_components = k, random_state = 42, init_params='random_from_data', reg_covar=1e-2)
    model.fit(data_used)
    print(model.score(data_used), "\n\n")
    ll_sk.append(model.score(data_used))
    # print(model.predict(data_used), "\n\n")
    # print(model.means_, "\n\n")
    # print(model.covariances_, "\n\n")
    # print(model.predict_proba(data_used), "\n\n")
    # print(model.weights_)

fig = sp.make_subplots(rows=1, cols=2)
fig.add_trace(go.Scatter(x = list(range(1, 10)), y = ll_self, mode = 'lines+markers', name = 'GMM_self'), row=1, col=1)
fig.add_trace(go.Scatter(x = list(range(1, 10)), y = ll_sk, mode = 'lines+markers', name = 'GMM_Sklearn'), row=1, col=2)
fig.update_xaxes(title_text = "K")
fig.update_yaxes(title_text = "Log Likelihood")
fig.update_layout(title_text = "Log Likelihood vs K for GMM")
fig.show()


########################## PCA ##########################
# data = data.drop(columns=['words'])
# data_used = data.drop('color', axis=1)
# model = PCA(n_components = 2)
# model.fit(data)
# print(model.transform().columns)

# #plot
# fig = px.scatter(x = model._data_trans[0], y = model._data_trans[1])
# fig.update_layout(title_text = "PCA")
# fig.show()


########################## Heirarchical Clustering ##########################

# # data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")
# # data_used = data.drop(columns=['words'])

# data = pd.read_csv(os.path.join(RawDataDIR, "data.csv")) #test dataset from Kaggle
# data_used = data.drop('color', axis=1)

# # pca = PCA(2)
# # pca.fit(data_used)
# # data_used = pca.transform()

# Z = linkage(data_used, 'ward')

# plt.figure(figsize=(25, 10))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
# plt.show()


