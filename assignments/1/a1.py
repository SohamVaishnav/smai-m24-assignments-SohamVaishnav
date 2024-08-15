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

def Dataloader(DataDIR: str, datafilename: str, model: str):
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

def Datasplitter(train: float, splitValid: bool, test: float, data: pd.DataFrame, model: str, class_feature: str):
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
                train_set = data_sorted[j:count[i]+j].sample(frac = train, random_state = 42)
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

model = KNN()
data = Dataloader(RawDataDIR, 'spotify.csv', 'KNN')
# train_set, valid_set, test_set = Datasplitter(0.8, False, 0.2, data, 'KNN', 'track_genre')
train_set, test_set = Datasplitter(0.8, False, 0.2, data, 'KNN', 'track_genre')

# print(train_set.shape, valid_set.shape, test_set.shape)
print(train_set.shape, test_set.shape)