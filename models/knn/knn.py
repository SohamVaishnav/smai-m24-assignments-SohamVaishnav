import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import importlib
import os


class KNN:
    ''' 
    This class contains all the functions related to K-Nearest Neighbors classification algorithm. To
    run the algorithm over a dataset, create a model in the main code by calling the class and running 
    the 'train', 'eval' and 'test' functions as and when required.
    '''
    def __init__(self) -> None:
        pass

    def SetNumNeighbors(self, k: int) -> None:
        ''' 
        Enter the number of neighbors (value of k) for KNN.

        Arguments: 
        k = integer value for the number of neighbors.
        '''
        self.k = k
        return None
    
    def SetDistMetric(self, dist_metric: str) -> None:
        ''' 
        Enter the distance metric that the model will use for computing loss funtion. \n
        Note: The model has been designed to evaluate and test based on only L1 and L2 distance 
        metrics.

        Arguments:
        dist_metric = string value denoting the distance metric to be used.
        '''
        self.dist_metric = dist_metric
        return None

    def GetDistMetric(self) -> str:
        ''' 
        Returns the distance metric that the model is currently using.
        '''
        return self.dist_metric
    
    def GetNumNeighbors(self) -> int:
        ''' 
        Returns the number of neighbors that are being currently considered by the model for
        classification.
        '''
        return self.k
    
    def train(self, X_train, y_train) -> None:
        ''' 
        Trains the model for classification tasks. (No concept of epochs for KNN)

        Arguments:
        X_train = A dataframe containing the data points and their features for training.
        y_train = A dataframe containing the labels corresponding to the X_train being used.
        '''
        self._X_train = X_train
        self._y_train = y_train

        return None
    
    def eval(self, X_valid, y_valid):
        ''' 
        Evaluates the performance of the model on validation set using the hyperparameters set so far. \n
        The computation renders a set of performace metrics for better insights. 

        Arguments:
        X_valid = A dataframe containing the data points and their features for validation.
        y_valid = A dataframe containing the labels corresponding to the X_valid being used.
        '''
        rows = X_valid.shape[0]
        cols = self._X_train.shape[1]

        dist_matrix = np.row_stack([np.sqrt(np.sum((X_valid[i] - self._X_train)**2, axis = 1) 
                                            for i in range(X_valid.shape[0]))])
        #dist_matrix is a matrix containing the distance of all the points in the validation set from
        #all the points in the train set.
        #the rows of the dist_matrix correspond to the datapoints in the validation set and the
        #columns denote the datapoints in the train set.

        #Testing is shown at the end of this file.

        dist_matrix = np.sort(dist_matrix, axis = 1) #sorting to find the minimum k distances
        dist_matrix = dist_matrix[:,0:self.k+1] #selecting the first k columns

        #need to note that while sorting, the index of the sorted datapoints in the original train set
        #must remain accessable else the labels won't be known and it will create whole lot of another
        #set of problems

        for i in range(dist_matrix.shape[0]):
            #write the code for selecting and applying the majority label out of the k closest datapoints
            #for each of the rows of the dist_matrix
            print("yet to be done...")
        
        return
    
    def test(self, X_test, y_test):
        ''' 
        Tests the performance of the model only once using the set hyperparameters derived from evaluation 
        and returns a set of performance measures to gain insights.

        Arguments:
        X_test = A dataframe containing the data points and their features for testing.
        y_test = A dataframe containing the labels corresponding to the X_test being used.
        '''
        return

### Testing the distance calculations using numpy
# a = 2*np.ones((10, 3))
# b = np.ones((5, 3))
# b[0, 0] = b[2, 2] = b[3, 2] = b[4, 1] = 0
# c = np.row_stack([np.sum((a[x] - b)**2, axis = 1) for x in range(a.shape[0])])
# print(c, "\n\n")
# c = np.sort(c, axis = 1)
# print(c, "\n\n")
# print(c[:,0:3])


