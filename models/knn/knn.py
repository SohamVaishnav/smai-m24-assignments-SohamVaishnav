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
    
    def DataSplitter(self, train: float, splitValid: bool, test: float, data: pd.DataFrame, class_feature: str) -> pd.DataFrame:
        ''' 
        Splits the data into train, validation and test sets based on the user defined ratios.\n
        The split is done based on stratified sampling to sustain the diversity of data in all the 
        three sets. \n
        Note: If splitValid is set to True, enter the values of train and test ratios accordingly. The 
        function will not divide the testing set into two parts if train + test ratios = 1

        Arguments:
        train = the ratio of data that needs to be set aside as training set.
        splitValid = boolean value to denote whether the validation set is required or no.
        test = the ratio of data that needs to be set aside as testing set.
        data = the dataset that is being discussed.
        '''
        if (not splitValid and train+test != 1):
            print("The sum of train and test ratios does not equate to unity. Please enter valid ratios "
                  "or allow creation of validation set.")
            return None, None
        elif (splitValid and train+test == 1):
            print("The sum of train and test ratios equate to unity and thus, the valid set cannot be "
                  "created. Please enter appropriate ratios.")
            return None, None, None
    
        labels = data[class_feature]
        uniqs, count = np.unique(labels, return_counts = True)
        print("The labels are: \n", uniqs)
        uniform_class_distribution = 0
        for i in range(len(uniqs)-1):
            if (count[i] == count[i+1]):
                uniform_class_distribution += 1
            else:
                uniform_class_distribution = 0
        if (uniform_class_distribution == len(uniqs)-1):
            print("The classes are uniformly distributed across the data.")
        else:
            print("The classes are not uniformly distributed across the data.")

        labels_ratios = count/np.sum(count)
        label_ratios_train = labels_ratios*train*data.shape[0]
        label_ratios_test = labels_ratios*test*data.shape[0]
        train_set = test_set = pd.DataFrame()
        if (splitValid):
            valid = 1 - (train + test)
            label_ratios_valid = labels_ratios*valid*data.shape[0]
            valid_set = pd.DataFrame()

        data_sorted = data.sort_values(class_feature)
        j = 0
        for i in range(0, len(count)):
            if (i == 0):
                train_set = data_sorted[j:count[i]+j].sample(frac = train , random_state = 42)
                data_sorted_temp = data_sorted[j:count[i]+j].drop(train_set.index, axis=0)

                test_set = data_sorted_temp.sample(n = int(label_ratios_test[i]), random_state = 42)
                data_sorted_temp = data_sorted_temp.drop(test_set.index, axis=0)

            else:
                temp = data_sorted[j:count[i]+j].sample(frac = train, random_state = 42)
                data_sorted_temp = data_sorted[j:count[i]+j].drop(temp.index, axis=0)
                train_set = pd.concat([train_set, temp])

                temp = data_sorted_temp.sample(n = int(label_ratios_test[i]), random_state = 42)
                data_sorted_temp = data_sorted_temp.drop(temp.index, axis=0)
                test_set = pd.concat([test_set, temp])

            j += count[i] 
        
        if (splitValid):
            temp = pd.concat([train_set, test_set])
            valid_set = data.drop(temp.index, axis = 0)
            return train_set, valid_set, test_set
        
        return train_set, test_set
    
    def DataRefiner(self, dataset: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Performs preprocessing on the data like removing null and NaN, and duplicate datapoints as well.

        Arguments:
        dataset = the dataset that the user is dealing with.
        '''
        dataset = dataset.drop_duplicates(subset = ['track_id'], keep = 'first')
        dataset = dataset.dropna(axis = 0)
        dataset['duration_ms'] = dataset['duration_ms']/(60*1000)
        dataset = dataset.rename(columns = {'duration_ms':'duration_min'})
        print("Final shape of the Data: ", dataset.shape)

        return dataset
    
    def DataNormaliser(self, dataset: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Normalises the data across all features to ensure that no one feature is singularly given undue 
        importance given its larger magnitude.

        Arguments:
        dataset = the data that is being used for the task.
        '''
        for i in dataset.columns:
            if (type(dataset[i].iloc[0]) == str or dataset[i].iloc[0].dtype == bool):
                continue
            means = np.mean(dataset[i], axis = 0)
            vars = np.var(dataset[i], axis = 0)
            dataset[i] = (dataset[i] - means)/np.sqrt(vars)
    
        return dataset
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
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
    
    def FindDistances(self, X_valid: pd.DataFrame, mode: str) -> pd.DataFrame:
        ''' 
        Computes the distances of all the data points in the validation set from all the data points in the 
        training set.

        Arguments:
        X_valid = A dataframe containing the data points and their features for validation.
        mode = A string value denoting whether the model is in initial mode or not.
        '''
        self._mode = mode
        if (self.dist_metric == 'l2'):
            if (mode == 'initial'):
                dist_matrix = pd.DataFrame([np.sqrt(np.sum((X_valid.iloc[i] - self._X_train)**2, axis = 1)) 
                                            for i in range(X_valid.shape[0])], index = X_valid.index)

            else:
                dist_matrix = np.row_stack([np.sum((X_valid.values[x:x+1000, None] - self._X_train.values)**2, axis= 2) 
                                            for x in range(0, X_valid.shape[0], 1000)])
                dist_matrix = pd.DataFrame(np.sqrt(dist_matrix), index=X_valid.index, columns=self._X_train.index)

        elif (self.dist_metric == 'l1'):
            if (mode == 'initial'):
                dist_matrix = pd.DataFrame([np.sum(np.abs(X_valid.iloc[i] - self._X_train), axis = 1)
                                            for i in range(X_valid.shape[0])], index = X_valid.index)
            else:
                dist_matrix = np.row_stack([np.sum((X_valid.values[x:x+10, None] - self._X_train.values)**2, axis= 2) for x in range(0, X_valid.shape[0], 10)])
                dist_matrix = pd.DataFrame(dist_matrix, index=X_valid.index, columns=self._X_train.index)

        elif (self.dist_metric == 'cosine'):
            if (mode == 'initial'):
                dist_matrix = pd.DataFrame([1 - np.dot(self._X_train, X_valid.iloc[i])/(self._norm_X_train*np.linalg.norm(X_valid.iloc[i]))
                                        for i in range(X_valid.shape[0])], index = X_valid.index, columns = self._X_train.index)

            else:
                dist_matrix = np.row_stack([1 - np.dot(self._X_train.values, X_valid.values.T)/(self._norm_X_train*np.linalg.norm(X_valid))])
                dist_matrix = pd.DataFrame(dist_matrix.T, index=X_valid.index, columns=self._X_train.index)

        self._dist_matrix = dist_matrix
    
    def predict(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        ''' 
        Evaluates the performance of the model on validation set using the hyperparameters set so far. \n
        The computation renders a set of performace metrics for better insights. 

        Arguments:
        X_valid = A dataframe containing the data points and their features for validation.
        y_valid = A dataframe containing the labels corresponding to the X_valid being used.
        '''

        #dist_matrix is a matrix containing the distance of all the points in the validation set from
        #all the points in the train set.
        #the rows of the dist_matrix correspond to the datapoints in the validation set and the
        #columns denote the datapoints in the train set. Demo for this is shown at the end of this file.

        pred_labels = []
        if (self._mode == 'initial'):
            for i in range(self._dist_matrix.shape[0]):
                temp = self._dist_matrix.iloc[i].sort_values(axis = 0)
                temp = temp[0:self.k]
                uniqs, count = np.unique(self._y_train.loc[temp.index], return_counts = True)
                label = uniqs[np.argmax(count)]
                pred_labels.append(label)
        
        else:
            dist_sorted_indices = self._dist_matrix.apply(lambda i: i.nsmallest(self.k).index, axis=1)
            dist_sorted_indices = dist_sorted_indices.apply(lambda i: i.values)
            for i in range(dist_sorted_indices.shape[0]):
                temp = self._y_train.loc[dist_sorted_indices.iloc[i]]
                uniqs, count = np.unique(temp, return_counts = True)
                label = uniqs[np.argmax(count)]
                pred_labels.append(label)
            
        pred_labels = pd.DataFrame(pred_labels, index = X_test.index)
        
        self._Meas = Measures(pred_values = pred_labels, true_values = y_test, 
                             labels = np.unique(self._y_train))

        return pred_labels, self._Meas.accuracy()*100, self._Meas.precision()*100, self._Meas.recall()*100, self._Meas.f1_score()*100

### Testing the distance calculations using numpy
# a = 2*np.ones((10, 3))
# b = np.ones((5, 3))
# b[0, 0] = b[2, 2] = b[3, 2] = b[4, 1] = 0
# c = np.row_stack([np.sum((a[x] - b)**2, axis = 1) for x in range(a.shape[0])])
# print(c, "\n\n")
# c = np.sort(c, axis = 1)
# print(c, "\n\n")
# print(c[:,0:3])


