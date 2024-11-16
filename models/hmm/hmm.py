import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from scipy.stats import multivariate_normal

from hmmlearn import hmm
import joblib

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('kde.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class HMM(object):
    '''
    Hidden Markov Model 
    '''
    def __init__(self, n_states, n_epochs):
        self.n_states = n_states
        self.hmm = None
        self.n_epochs = n_epochs

    def buildHMM(self, data):
        '''
        Build the HMM

        Parameters:
            data: (ndarray) the data
        '''
        self.data = data
        self.hmm = hmm.GaussianHMM(n_components=self.n_states, covariance_type='diag', 
                                   n_iter=self.n_epochs, random_state=42)
    
    def fitHMM(self, path_to_save, model_name):
        '''
        Fit the HMM

        Parameters:
            path_to_save: (str) the path to save the model
            model_name: (str) the name of the model
        '''
        self.hmm.fit(self.data, lengths=[len(self.data)])
        if not os.path.exists(path_to_save): 
            os.makedirs(path_to_save)
        self.save_model(os.path.join(path_to_save, model_name))
        print("Model saved successfully at ", path_to_save, " as ", model_name)
    
    def predictHMM(self, data, models_path, digits_likelihood: dict, digit: str):
        '''
        Predict using the HMM

        Parameters:
            data: (ndarray) the data to be predicted
            models_path: (str) the path to the model
            digits_likelihood: (dict) the likelihood of each digit
            digit: (str) the digit to be predicted
        '''
        assert os.path.exists(models_path), "Model not found"
        for root, _, files in os.walk(models_path):
            if len(files) == 0:
                print(f"No models found for digit {digit}")
                return
            for file in files:
                model = joblib.load(os.path.join(root, file))
                digits_likelihood[file.split('.')[0][-1]] = model.score(data)
        pred = max(digits_likelihood, key=digits_likelihood.get)
        return digits_likelihood, pred
    
    def save_model(self, path):
        '''
        Save the model

        Parameters:
            path: (str) the path to save the model
        '''
        joblib.dump(self.hmm, path)
    
    def accuracy(self, pred, true):
        '''
        Calculate the accuracy

        Parameters:
            pred: (ndarray) the predicted values
            true: (ndarray) the true values
        '''
        return np.mean(pred == true)
