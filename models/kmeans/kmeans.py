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

    def __init__(self, k = 2, epochs = 10) -> None:
        ''' 
        This function is used to initialize the class.

        Parameters:
            k (int): The number of clusters
            epochs (int): The number of epochs for which the model is to be trained. (default = 10)
        '''
        self._k = k
        self._epochs = epochs
        self._data = None
        pass

    def setK(self, k: int) -> None:
        ''' 
        This function is used to set the number of clusters.

        Parameters:
            k (int): The number of clusters
        '''
        self._k = k

    def getK(self):
        ''' 
        This function is used to return the number of clusters.
        '''
        return self._k
    
    def InitCentroids(self) -> np.ndarray:
        ''' 
        This function is used to initialize the centroids of the clusters using KMeans++ method.
        '''
        centroids = pd.DataFrame()
        centroids = pd.concat([centroids, self._data.sample(n=1, axis=0, random_state=42)])
        temp = self._data.drop(centroids.iloc[0].name, axis=0)

        for i in range(1, self._k):
            dist = np.row_stack([np.sum((temp.values[:,None] - centroids[i-1:i:].values)**2, axis = 2)])
            dist = pd.DataFrame(dist, columns=centroids[i-1:i:].index, index=temp.index)
            dist = dist/np.sum(dist)
            next = dist.sample(n=1, axis=0, random_state = 42, weights=dist.values.flatten())
            centroids = pd.concat([centroids, temp.loc[next.index]])
            temp = temp.drop(next.index, axis=0)

        return centroids
    
    def getCentroids(self):
        ''' 
        This function is used to return the centroids of the clusters.
        '''
        self._centroids = []
        for i in range(self._k):
            temp = self._data[self._data['clusters'] == i].copy()
            temp.drop('clusters', axis=1, inplace=True)
            self._centroids.append(np.mean(temp.values, axis=0))
            # self._centroids = pd.concat([self._centroids, pd.DataFrame([np.mean(temp.values, axis=0)], 
            #                                                            columns=temp.columns, index=[i])])
        return None
    
    def fit(self) -> np.ndarray:
        ''' 
        This function is used to fit the model to the dataset by dividing it into k appropriate 
        clusters.

        Parameters:
            TrainData (pd.DataFrame): The training data on which the model is to be fit.
        '''
        centroids = self.InitCentroids()
        self._centroids = centroids
        for k in range(self._epochs):
            # print("k: ", k)
            # print("Centroids: ", self._data.head())
            dist = np.row_stack([np.sum((self._data.values[:,None] - centroids.values)**2, axis=2)])
            dist = pd.DataFrame(dist, columns=centroids.index, index=self._data.index)
            self._data['clusters'] = np.argmin(dist, axis=1)
            self.getCentroids()
            if (k != self._epochs-1):
                self._data = self._data.drop('clusters', axis=1)
        WCSS = self.getCost()
        self._clusters = self._data['clusters']
        self._data.drop('clusters', axis=1, inplace=True)
        self._centroids = np.array(self._centroids)
        return self._centroids, WCSS, self._clusters

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
        self._model_cost = 0
        for i in range(self._k):
            temp = self._data[self._data['clusters'] == i].copy()
            temp.drop('clusters', axis=1, inplace=True)
            cost = np.sum((temp.values - self._centroids[i])**2, axis = 1)
            self._model_cost += np.sum(cost)
        return self._model_cost