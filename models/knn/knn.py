import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('knn.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from performance_measures.confusion_matrix import Confusion_Matrix, Measures

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
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        ''' 
        Trains the model for classification tasks. (No concept of epochs for KNN)

        Arguments:
        X_train = A dataframe containing the data points and their features for training.
        y_train = A dataframe containing the labels corresponding to the X_train being used.
        '''
        self._X_train = X_train
        self._y_train = y_train
        self._norm_X_train = np.linalg.norm(self._X_train)

        return None
    
    def eval(self, X_valid: pd.DataFrame, y_valid: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Evaluates the performance of the model on validation set using the hyperparameters set so far. \n
        The computation renders a set of performace metrics for better insights. 

        Arguments:
        X_valid = A dataframe containing the data points and their features for validation.
        y_valid = A dataframe containing the labels corresponding to the X_valid being used.
        '''
        if (self.dist_metric == 'l2'):
            dist_matrix = pd.DataFrame([np.sqrt(np.sum((X_valid.iloc[i] - self._X_train)**2, axis = 1)) 
                                            for i in range(X_valid.shape[0])], index = X_valid.index)
        elif (self.dist_metric == 'l1'):
            dist_matrix = pd.DataFrame([np.sum(np.abs(X_valid.iloc[i] - self._X_train), axis = 1)
                                            for i in range(X_valid.shape[0])], index = X_valid.index)
        elif (self.dist_metric == 'cosine'):
            dist_matrix = pd.DataFrame([1 - np.dot(self._X_train, X_valid.iloc[i])/(self._norm_X_train*np.linalg.norm(X_valid.iloc[i]))
                                            for i in range(X_valid.shape[0])],
                                            index = X_valid.index, columns = self._X_train.index)

        #dist_matrix is a matrix containing the distance of all the points in the validation set from
        #all the points in the train set.
        #the rows of the dist_matrix correspond to the datapoints in the validation set and the
        #columns denote the datapoints in the train set. Demo for this is shown at the end of this file.

        pred_labels = []
        rights = 0
        for i in range(dist_matrix.shape[0]):
            temp = dist_matrix.iloc[i].sort_values(axis = 0)
            temp = temp[0:self.k]
            uniqs, count = np.unique(self._y_train.loc[temp.index], return_counts = True)
            label = uniqs[np.argmax(count)]
            if (label == y_valid.iloc[i]):
                rights += 1
            pred_labels.append(label)
        print(rights/dist_matrix.shape[0])
        pred_labels = pd.DataFrame(pred_labels, index = X_valid.index)
        
        self._Meas = Measures(pred_values = pred_labels, true_values = y_valid, 
                             labels = np.unique(self._y_train))
        print("Accuracy = ", self._Meas.accuracy()*100)
        print("P_mac:P_mic = ", self._Meas.precision())
        print("R_mac:R_mic = ", self._Meas.recall())
        print("f1_mac:f1_mic = ", self._Meas.f1_score())

        return pred_labels
    
    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Tests the performance of the model only once using the set hyperparameters derived from evaluation 
        and returns a set of performance measures to gain insights.

        Arguments:
        X_test = A dataframe containing the data points and their features for testing.
        y_test = A dataframe containing the labels corresponding to the X_test being used.
        '''
        if (self.dist_metric == 'l2'):
            dist_matrix = pd.DataFrame([np.sqrt(np.sum((X_test.iloc[i] - self._X_train)**2, axis = 1)) 
                                            for i in range(X_test.shape[0])], index = X_test.index)
        elif (self.dist_metric == 'l1'):
            dist_matrix = pd.DataFrame([np.sum(np.abs(X_test.iloc[i] - self._X_train), axis = 1)
                                            for i in range(X_test.shape[0])], index = X_test.index)
        elif (self.dist_metric == 'cosine'):
            dist_matrix = pd.DataFrame([1 - np.dot(self._X_train, X_test.iloc[i])/(self._norm_X_train*np.linalg.norm(X_test.iloc[i]))
                                            for i in range(X_test.shape[0])],
                                            index = X_test.index, columns = self._X_train.index)

        #dist_matrix is a matrix containing the distance of all the points in the validation set from
        #all the points in the train set.
        #the rows of the dist_matrix correspond to the datapoints in the validation set and the
        #columns denote the datapoints in the train set. Demo for this is shown at the end of this file.

        pred_labels = []
        for i in range(dist_matrix.shape[0]):
            temp = dist_matrix.iloc[i].sort_values(axis = 0)
            temp = temp[0:self.k]
            uniqs, count = np.unique(self._y_train.loc[temp.index], return_counts = True)
            max_index = 0
            for j in range(0, len(count)):
                if (count[j] > max_index):
                    max_index = count[j]
                    label = uniqs[j]
            pred_labels.append(label)
        pred_labels = pd.DataFrame(pred_labels, index = X_test.index)

        self._Meas = Measures(pred_values = pred_labels, true_values = y_test, 
                             labels = np.unique(self._y_train))
        print("Accuracy = ", self._Meas.accuracy())
        print("P_mac:P_mic = ", self._Meas.precision())
        print("R_mac:R_mic = ", self._Meas.recall())
        print("f1_mac:f1_mic = ", self._Meas.f1_score())

        return pred_labels

### Testing the distance calculations using numpy
# a = 2*np.ones((10, 3))
# b = np.ones((5, 3))
# b[0, 0] = b[2, 2] = b[3, 2] = b[4, 1] = 0
# c = np.row_stack([np.sum((a[x] - b)**2, axis = 1) for x in range(a.shape[0])])
# print(c, "\n\n")
# c = np.sort(c, axis = 1)
# print(c, "\n\n")
# print(c[:,0:3])


