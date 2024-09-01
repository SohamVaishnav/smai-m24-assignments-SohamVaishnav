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

    def __init__(self, k: int) -> None:
        ''' 
        This function is used to initialize the class.

        Parameters:
            k (int): The number of clusters
        '''
        self._k = k
        pass

    def getK(self):
        ''' 
        This function is used to return the number of clusters.
        '''
        return self._k
    
    def fit(self):
        ''' 
        This function is used to fit the model to the dataset by dividing it into k appropriate 
        clusters.
        '''
        pass

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
