import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
import pandas as pd
import sys
import os

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('pca.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class KMeansClustering():
    ''' 
    This class is used to perform K-Means Clustering on the given dataset.
    '''

    def __init__(self, k: int, epochs = 10) -> None:
        ''' 
        This function is used to initialize the class.

        Parameters:
            k (int): The number of clusters
            epochs (int): The number of epochs for which the model is to be trained. (default = 10)
        '''
        self._k = k
        self._epochs = epochs
        pass

    def getK(self):
        ''' 
        This function is used to return the number of clusters.
        '''
        return self._k
    
    def InitCentroids(self) -> np.ndarray:
        ''' 
        This function is used to initialize the centroids of the clusters using KMeans++ method.
        '''
        centroids = []
        indices = []
        centroids.append(self._train_data.sample(n=1, axis=0, random_state=42).drop('words', axis=1))
        indices.append(centroids[0].index[0])
        temp = self._train_data.drop(centroids[0].index, axis=0)
        temp.drop('words', axis=1, inplace=True)

        for i in range(1, self._k):
            dist = pd.DataFrame([np.sum((temp.iloc[j].values - centroids[i-1].values)**2, axis=1) for j in range(temp.shape[0])], 
                                columns=[centroids[i-1].index[0]])
            dist = dist/np.sum(dist)
            next = dist.sample(n=1, axis=0, random_state=42)
            centroids.append(temp.loc[next.index])
            indices.append(next.index[0])
            temp = temp.drop(next.index, axis=0)
        centroids = np.array(centroids)
        return centroids, indices
    
    def fit(self, Data: pd.DataFrame) -> np.ndarray:
        ''' 
        This function is used to fit the model to the dataset by dividing it into k appropriate 
        clusters.

        Parameters:
            TrainData (pd.DataFrame): The training data on which the model is to be fit.
        '''
        self._data = Data
        centroids, indices = self.InitCentroids()
        for i in range(self._epochs):
            pass
        return centroids

    def predict(self):
        ''' 
        This function is used to predict the cluster to which the sample belongs.
        '''
        pass

    def getCost(self):
        ''' 
        This function is used to return the cost of the model which is Within-Cluster-Sum-of-Squares
        (WCSS).
        '''
        self._mod_cost = None
        return self._mod_cost


# a = 2*np.ones((10, 3))
# b = np.ones((1, 3))
# b[0, 0] = b[2, 2] = b[3, 2] = b[4, 1] = 0
# c = np.row_stack([np.sum((a[x] - b)**2, axis = 1) for x in range(a.shape[0])])
# print(c, "\n\n")
# c = np.sort(c, axis = 1)
# print(c, "\n\n")
# print(c[:,0:3])