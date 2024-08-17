import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Confusion_Matrix:
    ''' 
    Creates and returns the confusion matrix by selecting True and False positives and negatives.
    '''
    def __init__(self) -> None:
        pass

    def SetModel(self, model: str) -> None:
        ''' 
        Initialises the class for the model that is being used.

        Arguments:
        model = string denoting the model being used.
        '''
        self._model = model
        return None
    
    def GetModel(self):
        '''
        Returns the model that is currently being taken into consideration.
        '''
        return self._model

    def FindTP(self, pred_vals, true_vals, labels) -> pd.DataFrame:
        ''' 
        Finds the true positives by comparing the predicted and true values. 

        Arguments:
        pred_vals = an array containing all the predicted values.
        true_vals = an array containing all the true values.
        labels = an array containing all the possible labels.
        '''
        labelwise_TP = []
        labels = np.sort(labels)
        uniq_predvals, count_pred = np.unique(pred_vals, return_counts = True)
        uniq_truevals, count_true = np.unique(true_vals, return_count = True)
        
            
        return 
    
    def FindTN(self) -> pd.DataFrame:
        return 
    
    def FindFN(self) -> pd.DataFrame:
        return 
    
    def FindTN(self) -> pd.DataFrame:
        return 
    
class Measures:
    ''' 
    Computes and returns the performance measures for the model based on the predicted and ground-
    truth values.
    '''
    def __init__(self) -> None:
        pass 