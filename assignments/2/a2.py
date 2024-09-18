import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.feather as feather
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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

def DataLoader_KNN(DataDIR: str, datafilename: str, model: str):
    ''' 
    Loads the data from the directory. \n
    Note: Currently this function only deals with reading .csv files.

    Arguments:
    DataDIR = string denoting the path to the directory containing data.
    datafilename = string denoting the name of the data file that needs to be read from DataDIR
    model = the model that the data is being loaded for.
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)
    assert os.path.exists(data_path), f"{data_path} path is invalid!"

    if datafilename.endswith('.csv'):
        if (model == 'KNN'):
            data = pd.read_csv(data_path, sep = ',', index_col = 0)
            print("Data has been read.")
            print("Data size: (rows, cols)", data.shape)
            return data
        elif (model == 'LinReg'):
            x, y = np.loadtxt(data_path, delimiter = ',', dtype = np.float64, unpack = True, skiprows = 1)
            print("Data has been unpacked and loaded.")
            print("X: ", x.shape)
            print("Y: ", y.shape)
            return x, y

########################## KMeans Clustering ##########################
# data = DataLoader(RawDataDIR, "word-embeddings.feather")
# print(data.head())

# data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")
# print(data.head())

# data = pd.read_csv(os.path.join(RawDataDIR, "data.csv")) #test dataset from Kaggle

# model = KMeansClustering(epochs = 10)
# # data_used = data.drop('color', axis=1)
# data_used = data.drop(columns=['words'])
# model._data = data_used.copy()
# err = []
# err_log = []
# for k in range(1, 50):
#     print(k)
#     model.setK(k)
#     centroids, WCSS, _ = model.fit()
#     err.append(WCSS)
#     err_log.append("WCSS for k = "+ str(k) +" is " + str(WCSS))

# # err_log = pd.DataFrame(err_log)
# # err_log.to_csv(os.path.join(CurrDIR, "err_logs/kmeans_wcss.csv"), index=False)

# err_sk = []
# err_log = []
# for i in range(1, 50):
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

# k2 = 4 #got from the plot
# model.setK(k2)
# centroids, WCSS, _ = model.fit()
# print("WCSS: ", WCSS)
# print(centroids)
# print("Clusters: ", model._clusters)


# pca = PCA(2)
# pca.fit(data_used)
# print(pca._eigvals)
# evr = pca.getExplainedVarRatio()

# fig = px.line(x = list(range(1, 21)), y = evr[0:20], markers=True)
# fig.update_layout(title_text = "Scree Plot")
# fig.show()

# pca = PCA(4)
# pca.fit(data_used)
# data_used = pca.transform()

# err = []
# err_log = []
# model._data = data_used.copy()
# for k in range(1, 15):
#     print(k)
#     model.setK(k)
#     centroids, WCSS, _ = model.fit()
#     err.append(WCSS)
#     err_log.append("WCSS for k = "+ str(k) +" is " + str(WCSS))

# fig = px.line(x = list(range(1, 15)), y = err, markers=True)
# fig.update_layout(title_text = "WCSS vs K for KMeans")
# fig.show()

# kmeans3 = 5
# model._data = data_used.copy()
# model.setK(kmeans3)
# centroids, WCSS, _ = model.fit()
# print("WCSS: ", WCSS)
# print(centroids)
# print("Clusters: ", model._clusters)

# fig = px.scatter_3d(x = data_used[0], y = data_used[1], z = data_used[2], color = model._clusters)
# fig.add_trace(go.Scatter3d(x = centroids[:, 0], y = centroids[:, 1], z = centroids[:, 2], mode = 'markers', marker = dict(size = 10)))
# fig.update_layout(title_text = "KMeans Clustering")
# fig.show()

# table = []
# for i in range(kmeans3):
#     temp = data['words'][model._clusters == i]
#     table.append(temp)
# table = pd.DataFrame(table, index=[0, 1, 2, 3, 4]).T
# print(table)
# table.to_markdown(os.path.join(CurrDIR, "kmeans_clusters.md"))

########################## GMM ##########################

model = GaussianMixtureModel()
data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")
data_used = data.drop(columns=['words'])
# data = pd.read_csv(os.path.join(RawDataDIR, "data.csv")) #test dataset from Kaggle
# data_used = data.drop('color', axis=1)

pca = PCA(4)
pca.fit(data_used)
data_used = pca.transform()

# ll_self = []
# AIC_self = []
# BIC_self = []
# for k in range(1, 10):
#     print(k)
#     model.fit(data_used, K = k, init_method = "random_from_data", epochs = 4, epsilon = 1e-6)
#     model.getLikelihood()
#     ll_self.append(model._likelihood)
#     AIC_self.append(model.doAIC())
#     BIC_self.append(model.doBIC())
    # print(model.getMembership(epsilon=1e-2).shape, "\n\n")
    # print(model.getCluster())
    # print(model.getParams()[2])

# #plot AIC
# fig = px.line(x = list(range(1, 10)), y = AIC_self, markers=True)
# fig.update_layout(title_text = "AIC vs K for GMM")
# fig.show()

# table = []
# for i in range(4):
#     temp = data['words'][model.getCluster() == i]
#     table.append(temp)
# table = pd.DataFrame(table, index=[0, 1, 2, 3]).T
# print(table)
# # store as markdown file
# table.to_markdown(os.path.join(CurrDIR, "gmm_clusters.md"))


#plot clusters for reduced data in 3D
# fig = px.scatter_3d(x = data_used[0], y = data_used[1], z = data_used[2], color = model.getCluster())
# fig.update_layout(title_text = "GMM")
# fig.show()

ll_sk = []
AIC_sk = []
BIC_sk = []
for k in range(1, 10):
    model = GaussianMixture(n_components = k, random_state = 42, init_params='random_from_data', reg_covar=1e-5)
    model.fit(data_used)
    print(model.score(data_used)*200, "\n\n")
    ll_sk.append(model.score(data_used)*200)
    AIC_sk.append(model.aic(data_used))
    BIC_sk.append(model.bic(data_used))
    # print(model.predict(data_used), "\n\n")
    # print(model.means_, "\n\n")
    # print(model.covariances_, "\n\n")
    # print(model.predict_proba(data_used), "\n\n")
    # print(model.weights_)

fig = px.line(x = list(range(1, 10)), y = AIC_sk, markers=True)
fig.update_layout(title_text = "AIC vs K for GMM")
fig.show()

# fig = sp.make_subplots(rows=1, cols=2)
# fig.add_trace(go.Scatter(x = list(range(1, 10)), y = ll_self, mode = 'lines+markers', name = 'GMM_self'), row=1, col=1)
# fig.add_trace(go.Scatter(x = list(range(1, 10)), y = ll_sk, mode = 'lines+markers', name = 'GMM_Sklearn'), row=1, col=2)
# fig.update_xaxes(title_text = "K")
# fig.update_yaxes(title_text = "Log Likelihood")
# fig.update_layout(title_text = "Log Likelihood vs K for GMM")
# fig.show()

# fig = sp.make_subplots(rows=1, cols=2)
# fig.add_trace(go.Scatter(x = list(range(1, 10)), y = AIC_self, mode = 'lines+markers', name = 'GMM_self'), row=1, col=1)
# fig.add_trace(go.Scatter(x = list(range(1, 10)), y = AIC_sk, mode = 'lines+markers', name = 'GMM_Sklearn'), row=1, col=2)
# fig.update_xaxes(title_text = "K")
# fig.update_yaxes(title_text = "AIC")
# fig.update_layout(title_text = "AIC vs K for GMM")
# fig.show()

# fig = sp.make_subplots(rows=1, cols=2)
# fig.add_trace(go.Scatter(x = list(range(1, 10)), y = BIC_self, mode = 'lines+markers', name = 'GMM_self'), row=1, col=1)
# fig.add_trace(go.Scatter(x = list(range(1, 10)), y = BIC_sk, mode = 'lines+markers', name = 'GMM_Sklearn'), row=1, col=2)
# fig.update_xaxes(title_text = "K")
# fig.update_yaxes(title_text = "BIC")
# fig.update_layout(title_text = "BIC vs K for GMM")
# fig.show()


########################## PCA ##########################
# data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")
# data_used = data.drop(columns=['words'])

# model = PCA(n_components = 2)
# model.fit(data_used)
# model.transform()
# print(model.checkPCA())

# fig = px.scatter(x = model._data_trans[0], y = model._data_trans[1], color = data['words'], labels = {'x': 'PC1', 'y': 'PC2'})
# fig.update_layout(title_text = "PCA")
# fig.show()

# model = PCA(n_components = 3)
# model.fit(data_used)
# model.transform()
# print(model.checkPCA())

#plot 3D data
# fig = px.scatter_3d(x = model._data_trans[0], y = model._data_trans[1], z = model._data_trans[2], color = data['words'], labels = {'x': 'PC1', 'y': 'PC2', 'z': 'PC3'})
# fig.update_layout(title_text = "PCA")
# fig.show()

# #plot
# fig = px.scatter(x = model._data_trans[0], y = model._data_trans[1])
# fig.update_layout(title_text = "PCA")
# fig.show()


########################## Heirarchical Clustering ##########################

# data = DataLoader(PreProcessDIR, "word-embeddings_v1.feather")
# data_used = data.drop(columns=['words'])

# data = pd.read_csv(os.path.join(RawDataDIR, "data.csv")) #test dataset from Kaggle
# data_used = data.drop('color', axis=1)

# # pca = PCA(2)
# # pca.fit(data_used)
# # data_used = pca.transform()

# Z1 = linkage(data_used, 'ward', 'euclidean')
# Z2 = linkage(data_used, 'single', 'euclidean')
# Z3 = linkage(data_used, 'complete', 'euclidean')
# Z4 = linkage(data_used, 'average', 'euclidean')

# fig, ax = plt.subplots(2, 2, figsize=(25, 15))
# fig.suptitle('Hierarchical Clustering Dendrogram | Euclidean Distance')
# dendrogram(Z1, ax=ax[0, 0])
# ax[0, 0].set_title('Ward')
# dendrogram(Z2, ax=ax[0, 1])
# ax[0, 1].set_title('Single')
# dendrogram(Z3, ax=ax[1, 0])
# ax[1, 0].set_title('Complete')
# dendrogram(Z4, ax=ax[1, 1])
# ax[1, 1].set_title('Average')
# plt.show()


# Z2 = linkage(data_used, 'single', 'cityblock')
# Z3 = linkage(data_used, 'complete', 'cityblock')
# Z4 = linkage(data_used, 'average', 'cityblock')

# fig, ax = plt.subplots(3, 1, figsize=(15, 15))
# fig.suptitle('Hierarchical Clustering Dendrogram | Cityblock Distance')
# dendrogram(Z2, ax=ax[0])
# ax[0].set_title('Single')
# dendrogram(Z3, ax=ax[1])
# ax[1].set_title('Complete')
# dendrogram(Z4, ax=ax[2])
# ax[2].set_title('Average')
# plt.show()


# Z2 = linkage(data_used, 'single', 'cosine')
# Z3 = linkage(data_used, 'complete', 'cosine')
# Z4 = linkage(data_used, 'average', 'cosine')

# fig, ax = plt.subplots(3, 1, figsize=(15, 15))
# fig.suptitle('Hierarchical Clustering Dendrogram | Cosine Distance')
# dendrogram(Z2, ax=ax[0])
# ax[0].set_title('Single')
# dendrogram(Z3, ax=ax[1])
# ax[1].set_title('Complete')
# dendrogram(Z4, ax=ax[2])
# ax[2].set_title('Average')
# plt.show()


# Z2 = linkage(data_used, 'single', 'correlation')
# Z3 = linkage(data_used, 'complete', 'correlation')
# Z4 = linkage(data_used, 'average', 'correlation')

# fig, ax = plt.subplots(3, 1, figsize=(15, 15))
# fig.suptitle('Hierarchical Clustering Dendrogram | Correlation Distance')
# dendrogram(Z2, ax=ax[0])
# ax[0].set_title('Single')
# dendrogram(Z3, ax=ax[1])
# ax[1].set_title('Complete')
# dendrogram(Z4, ax=ax[2])
# ax[2].set_title('Average')
# plt.show()


# k = 5 #(kmeans)
# clusters = fcluster(Z1, k, criterion='maxclust')
# data['clusters'] = clusters
# #convert into a markdown table with clusters number as columns
# table = []
# for i in range(k):
#     temp = data['words'][data['clusters'] == i+1]
#     table.append(temp)
# table = pd.DataFrame(table, index=[0, 1, 2, 3, 4]).T
# table.to_markdown(os.path.join(CurrDIR, "hierarchical_clusters_kmeans.md"))

# k = 4 #(gmm)
# clusters = fcluster(Z1, k, criterion='maxclust')
# data['clusters'] = clusters
# #convert into a markdown table with clusters number as columns
# table = []
# for i in range(k):
#     temp = data['words'][data['clusters'] == i+1]
#     table.append(temp)
# table = pd.DataFrame(table, index=[0, 1, 2, 3]).T
# table.to_markdown(os.path.join(CurrDIR, "hierarchical_clusters_gmm.md"))


# k = 5 #(kmeans)
# clusters = fcluster(Z2, k, criterion='maxclust')
# data['clusters'] = clusters
# #convert into a markdown table with clusters number as columns
# table = []
# for i in range(k):
#     temp = data['words'][data['clusters'] == i+1]
#     table.append(temp)
# table = pd.DataFrame(table, index=[0, 1, 2, 3, 4]).T
# table.to_markdown(os.path.join(CurrDIR, "hierarchical_clusters_single.md"))

# k = 5 #(kmeans)
# clusters = fcluster(Z3, k, criterion='maxclust')
# data['clusters'] = clusters
# #convert into a markdown table with clusters number as columns
# table = []
# for i in range(k):
#     temp = data['words'][data['clusters'] == i+1]
#     table.append(temp)
# table = pd.DataFrame(table, index=[0, 1, 2, 3, 4]).T
# table.to_markdown(os.path.join(CurrDIR, "hierarchical_clusters_collective.md"))

# k = 5 #(kmeans)
# clusters = fcluster(Z4, k, criterion='maxclust')
# data['clusters'] = clusters
# #convert into a markdown table with clusters number as columns
# table = []
# for i in range(k):
#     temp = data['words'][data['clusters'] == i+1]
#     table.append(temp)
# table = pd.DataFrame(table, index=[0, 1, 2, 3, 4]).T
# table.to_markdown(os.path.join(CurrDIR, "hierarchical_clusters_average.md"))


########################## KNN ##########################
# from models.knn.knn import KNN

# PreProcessDIR = os.path.join(UserDIR, "./data/interim/1/spotify_KNN/")

# model = KNN()
# isValid = True
# data = DataLoader_KNN(PreProcessDIR, 'spotify_word2num.csv', 'KNN')
# data = model.DataRefiner(data)
# string_features = ['track_id', 'artists', 'album_name', 'track_name', 'explicit']
# data = Word2Num(data, string_features, 'KNN')
# data = model.DataNormaliser(data)
# data_test = data.drop(columns = ['track_genre'])

# pca = PCA(4)
# pca.fit(data_test)
# evr = pca.getExplainedVarRatio()
# data_used = pca.transform()

# fig = px.line(x = list(range(1, len(evr)+1)), y = evr, markers=True)
# fig.update_layout(title_text = "Scree Plot for Spotify")
# fig.show()

# data_used['track_genre'] = data['track_genre']

# train_set, valid_set, test_set = model.DataSplitter(0.8, isValid, 0.1, data_used, 'track_genre')
# print("Training set: ", train_set.shape)
# print("Testing set: ", test_set.shape)
# print("Validation set: ", valid_set.shape)

# y_train = train_set['track_genre']
# X_train = train_set.drop(columns = ['track_genre'], axis = 1)

# y_test = test_set['track_genre']
# X_test = test_set.drop(columns = ['track_genre'], axis = 1)

# y_valid = valid_set['track_genre']
# X_valid = valid_set.drop(columns = ['track_genre'], axis = 1)

# time_start = time.time()
# model.fit(X_train, y_train)
# model.SetDistMetric('l1')
# model.SetNumNeighbors(15)
# model.FindDistances(X_test, 'optimised')
# y_pred, acc, prec, recall, f1 = model.predict(X_test, y_test)
# time_end = time.time()
# print("Accuracy: ", acc)
# print("Precision: ", prec)
# print("Recall: ", recall)
# print("F1 Score: ", f1)
# print("Time taken: ", time_end - time_start)


