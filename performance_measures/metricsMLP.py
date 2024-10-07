import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class Confusion_Matrix:
    ''' 
    Creates and returns the confusion matrix by selecting True and False positives and negatives.
    '''
    def __init__(self, isMulti = False) -> None:
        '''
        Initialises the class for the model that is being used.

        Arguments:
        isMulti = boolean denoting whether the model is a multi-class model or not
        '''
        self._isMulti = isMulti
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
    
    def CreateMatrix(self, pred_vals, true_vals, labels, labels_isnum: False) -> None:
        ''' 
        Creates a matrix for classwise distribution of predicted and true values.

        Arguments:
        pred_vals = the array containing predicted values
        true_vals = the array containing true values
        labels = the labels for the classes
        labels_isnum = boolean denoting whether the labels are numerical or not
        '''
        self._labels = labels
        self._pred_vals = pred_vals.astype(int)
        self._true_vals = true_vals.astype(int)

        if (not self._isMulti):
            if (not labels_isnum):
                ConfMat = np.zeros((len(labels), len(labels)))
                labels = {labels[i]:i for i in range(len(labels))}

                pred_vals = [labels[pred_vals[i]] for i in range(len(pred_vals))]
                true_vals = [labels[true_vals[i]] for i in range(len(true_vals))]

            else:
                pred_vals = np.argmax(pred_vals, axis = 1)
                true_vals = np.argmax(true_vals, axis = 1)
                ConfMat = np.zeros((max(labels)+1, max(labels)+1))

            for i in range(true_vals.shape[0]):
                ConfMat[pred_vals[i], true_vals[i]] += 1

            self._ConfMat = ConfMat
        return None


    def FindTP(self) -> dict:
        ''' 
        Finds the true positives by comparing the predicted and true values. 
        '''
        if (self._isMulti):
            self._TP = np.sum(self._pred_vals & self._true_vals, axis = 0)
            return self._TP

        labelwise_TP = {i:self._ConfMat[i, i] for i in range(self._ConfMat.shape[0])}
        self._TP = labelwise_TP
        return labelwise_TP
    
    def FindTN(self) -> dict:
        '''
        Finds the true negatives by comparing the predicted and true values.
        '''
        if (self._isMulti):
            self._TN = np.sum((~self._pred_vals) & (~self._true_vals), axis = 0)
            return self._TN
        
        labelwise_TN = {}
        for i in range(self._ConfMat.shape[0]):
            sum = self._ConfMat[:i].sum() + self._ConfMat[i+1:].sum()
            labelwise_TN.update({i:sum})

        self._TN = labelwise_TN
        return labelwise_TN
    
    def FindFN(self) -> dict:
        ''' 
        Finds the false negatives by comparing the predicted and true values. 
        '''
        if (self._isMulti):
            self._FN = np.sum((~self._pred_vals) & self._true_vals, axis = 0)
            return self._FN
        
        labelwise_FN = {}
        for i in range(self._ConfMat.shape[1]):
            sum = self._ConfMat[:,i].sum() - self._ConfMat[i, i]
            labelwise_FN.update({i:sum})

        self._FN = labelwise_FN
        return labelwise_FN
    
    def FindFP(self) -> dict:
        ''' 
        Finds the false positives by comparing the predicted and true values. 
        '''
        if (self._isMulti):
            self._FP = np.sum(self._pred_vals & (~self._true_vals), axis = 0)
            return self._FP
        
        labelwise_FP = {}
        for i in range(self._ConfMat.shape[0]):
            sum = self._ConfMat[i].sum() - self._ConfMat[i, i]
            labelwise_FP.update({i:sum})

        self._FP = labelwise_FP
        return labelwise_FP
    
    
class Measures():
    ''' 
    Computes and returns the performance measures for the model based on the predicted and ground-
    truth values.
    '''
    def __init__(self, pred_values, true_values, labels, labels_isnum = False, isMulti = False) -> None:
        '''
        Initialises the class for the model that is being used.

        Arguments:
        pred_values = the array containing predicted values
        true_values = the array containing true values
        labels = the labels for the classes
        labels_isnum = boolean denoting whether the labels are numerical or not
        isMulti = boolean denoting whether the model is a multi-class model or not
        '''
        self._isMulti = isMulti
        self._pred_vals = pred_values
        self._true_vals = true_values
        self._labels = labels
        self._CM = Confusion_Matrix(isMulti)
        self._CM.CreateMatrix(pred_values, true_values, labels, labels_isnum)
        self._CM.FindTP()
        self._CM.FindTN()   
        self._CM.FindFP()
        self._CM.FindFN()
        pass 

    def precision(self) -> float:
        ''' 
        Computes precision scores - mirco and macro - for the model.
        '''
        if (self._isMulti):
            self._precision_macro = np.mean(self._CM._TP/(self._CM._TP + self._CM._FP))
            self._precision_micro = np.sum(self._CM._TP)/(np.sum(self._CM._TP + self._CM._FP))
            return self._precision_macro, self._precision_micro
        
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
        if (self._isMulti):
            self._recall_macro = np.mean(self._CM._TP/(self._CM._TP + self._CM._FN))
            self._recall_micro = np.sum(self._CM._TP)/(np.sum(self._CM._TP) + np.sum(self._CM._FN))
            return self._recall_macro, self._recall_micro
        
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
        if (self._isMulti):
            self._f1_macro = 2*self._precision_macro*self._recall_macro/(self._precision_macro+self._recall_macro)
            self._f1_micro = 2*self._precision_micro*self._recall_micro/(self._precision_micro+self._recall_micro)
            return self._f1_macro, self._f1_micro
        
        f1_macro = 2*self._pred_macro*self._recall_macro/(self._pred_macro+self._recall_macro)
        f1_micro = 2*self._pred_micro*self._recall_micro/(self._pred_micro+self._recall_micro)

        self._f1_macro = f1_macro
        self._f1_micro = f1_micro
        return f1_macro, f1_micro
    
    def accuracy(self) -> float:
        ''' 
        Computes model accuracy.
        '''
        if (self._isMulti):
            self._hard_acc = np.all(self._pred_vals == self._true_vals, axis = 1)
            self._hard_acc = np.mean(self._hard_acc)
            self._soft_acc = sum(self._CM._TP)/(np.sum(self._true_vals))
            return self._soft_acc, self._hard_acc
        
        Acc = sum(self._CM._TP.values())/(self._CM._ConfMat.sum())
        self._acc = Acc
        return Acc