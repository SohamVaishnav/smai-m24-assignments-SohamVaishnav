import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
    
    def CreateMatrix(self, pred_vals: pd.DataFrame, true_vals: pd.DataFrame, labels) -> None:
        ''' 
        Creates a matrix for classwise distribution of predicted and true values.

        Arguments:
        pred_vals = the array containing predicted values
        true_vals = the array containing true values
        '''
        ConfMat = pd.DataFrame(index = labels, columns = labels)
        ConfMat.iloc[:] = 0
        
        for i in true_vals.index:
            ConfMat.loc[pred_vals.loc[i], true_vals.loc[i]] += 1

        self._ConfMat = ConfMat

        # print(self._ConfMat.index, "\n\n")
        # print(self._ConfMat.columns, "\n\n")
        return None


    def FindTP(self) -> dict:
        ''' 
        Finds the true positives by comparing the predicted and true values. 
        '''
        labelwise_TP = {i:self._ConfMat.loc[i, i] for i in self._ConfMat.index}
        self._TP = labelwise_TP
        return labelwise_TP
    
    def FindTN(self) -> dict:
        '''
        Finds the true negatives by comparing the predicted and true values.
        '''
        labelwise_TN = {}
        for i in self._ConfMat.index:
            temp = self._ConfMat.drop(i, axis = 1)
            temp = temp.drop(i, axis = 0)
            sum = temp.to_numpy().sum()
            labelwise_TN.update({i:sum})

        self._TN = labelwise_TN
        return labelwise_TN
    
    def FindFN(self) -> dict:
        ''' 
        Finds the false negatives by comparing the predicted and true values. 
        '''
        labelwise_FN = {}
        for i in self._ConfMat.columns:
            sum = self._ConfMat[i].to_numpy().sum() - self._ConfMat.loc[i, i]
            labelwise_FN.update({i:sum})

        self._FN = labelwise_FN
        return labelwise_FN
    
    def FindFP(self) -> dict:
        ''' 
        Finds the false positives by comparing the predicted and true values. 
        '''
        labelwise_FP = {}
        for i in self._ConfMat.index:
            sum = self._ConfMat.loc[i].to_numpy().sum() - self._ConfMat.loc[i, i]
            labelwise_FP.update({i:sum})

        self._FP = labelwise_FP
        return labelwise_FP
    
class Measures():
    ''' 
    Computes and returns the performance measures for the model based on the predicted and ground-
    truth values.
    '''
    def __init__(self, pred_values: pd.DataFrame, true_values: pd.DataFrame, labels) -> None:
        self._CM = Confusion_Matrix()
        self._CM.CreateMatrix(pred_values, true_values, labels)
        self._CM.FindFN()
        self._CM.FindFP()
        self._CM.FindTN()
        self._CM.FindTP()
        pass 

    def precision(self) -> float:
        ''' 
        Computes precision scores - mirco and macro - for the model.
        '''
        temp = [self._CM._TP,  
                self._CM._FP]
        denom = Counter()
        for score in temp:
            denom.update(score)
        denom = dict(denom)
        denom = list(denom.values())
        numer = list(self._CM._TP.values())
        pred_macro = 0
        for i in range(len(denom)):
            if (denom[i] != 0):
                pred_macro += numer[i]/denom[i]
        pred_macro /= self._CM._ConfMat.shape[0]
        self._pred_macro = pred_macro

        pred_micro = sum(self._CM._TP.values())/(sum(self._CM._TP.values()) + sum(self._CM._FP.values()))
        self._pred_micro = pred_micro
        return pred_macro, pred_micro
    
    def recall(self) -> float:
        ''' 
        Computes recall scores - micro and macro - for the model.
        '''
        temp = [self._CM._TP,  
                self._CM._FN]
        denom = Counter()
        for score in temp:
            denom.update(score)
        denom = dict(denom)
        denom = list(denom.values())
        numer = list(self._CM._TP.values())
        recall_macro = 0
        for i in range(len(denom)):
            if (denom[i] != 0):
                recall_macro += numer[i]/denom[i]
        recall_macro /= self._CM._ConfMat.shape[0]
        self._recall_macro = recall_macro

        recall_micro = sum(self._CM._TP.values())/(sum(self._CM._TP.values()) + sum(self._CM._FN.values()))
        self._recall_micro = recall_micro
        return recall_macro, recall_micro
    
    def f1_score(self) -> float:
        ''' 
        Computes f1 scores - micro and macro - for the model.
        '''
        f1_macro = 2*self._pred_macro*self._recall_macro/(self._pred_macro+self._recall_macro)
        f1_micro = 2*self._pred_micro*self._recall_micro/(self._pred_micro+self._recall_micro)

        self._f1_macro = f1_macro
        self._f1_micro = f1_micro
        return f1_macro, f1_micro
    
    def accuracy(self) -> float:
        ''' 
        Computes model accuracy.
        '''
        Acc = sum(self._CM._TP.values())/(self._CM._ConfMat.to_numpy().sum())
        self._acc = Acc
        return Acc