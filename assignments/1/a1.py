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

def DataLoader(DataDIR: str, datafilename: str, model: str):
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
        elif (model == 'LinReg'):
            data = pd.read_csv(data_path, sep = ',')
    
    #printing data properties
    print("Data has been read.")
    print("Data size: (rows, cols)", data.shape)

    return data

def DataWriter(DataDIR: str, datafilename: str, extension: str, data: pd.DataFrame) -> str:
    '''
    Writes data in the desired format (extension) at the given location having the desired name.\n
    Note: The function currently handles only .csv files.

    Arguments:
    DataDIR = the directory where data will be written into
    datafilename = the name of the file of data which will be written
    extension = the format of storage (e.g: .csv)
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)
    data.to_csv(data_path, index = True)
    
    return data_path

def DataSplitter(train: float, splitValid: bool, test: float, data: pd.DataFrame, model: str, class_feature: str) -> pd.DataFrame:
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
    model = string denoting the model for which the data is being splitted.
    '''
    if (not splitValid and train+test != 1):
        print("The sum of train and test ratios does not equate to unity. Please enter valid ratios "
              "or allow creation of validation set.")
        return None, None
    elif (splitValid and train+test == 1):
        print("The sum of train and test ratios equate to unity and thus, the valid set cannot be "
              "created. Please enter appropriate ratios.")
        return None, None, None
    
    if (model == 'KNN'):
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

def DataRefiner(dataset: pd.DataFrame,  model: str) -> pd.DataFrame:
    ''' 
    Performs preprocessing on the data like removing null and NaN, and duplicate datapoints as well.

    Arguments:
    dataset = the dataset that the user is dealing with.
    model = the kind of model that the data is being preprocessed for.
    '''
    if (model == 'KNN'):
        dataset = dataset.drop_duplicates(subset = ['track_id'], keep = 'first')
        dataset = dataset.dropna(axis = 0)
        print("Final shape of the Data: ", dataset.shape)

    return dataset
    
model = KNN()
isValid = True
data = DataLoader(RawDataDIR, 'spotify.csv', 'KNN')
data = DataRefiner(data, 'KNN')
train_set, valid_set, test_set = DataSplitter(0.7, isValid, 0.2, data, 'KNN', 'track_genre')
print("Training set: ", train_set.shape)
print("Testing set: ", test_set.shape)
print("Validation set: ", valid_set.shape)

KNN_PreProcessDIR = os.path.join(PreProcessDIR, 'spotify_KNN/')
DataWriter(KNN_PreProcessDIR, 'train_set_refined.csv', '.csv', train_set)
DataWriter(KNN_PreProcessDIR, 'test_set_refined.csv', '.csv', test_set)
DataWriter(KNN_PreProcessDIR, 'valid_set_refined.csv', '.csv', valid_set)