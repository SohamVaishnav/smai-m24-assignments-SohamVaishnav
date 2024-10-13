import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('auto_encoder.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.MLP.auto_encoder import *
from models.MLP.mlp_regression import *

class AutoEncoder(object):
    ''' 
    AutoEncoder is a class that implements an autoencoder for dimensionality reduction.
    '''
    def __init__(self, config) -> None:
        ''' 
        Initializes the AutoEncoder class.
        '''
        self._model = MutliLayerPerceptron_Regression(config)
        self._model.add()
    
    def fit(self, X, y, X_valid, y_valid):
        ''' 
        Fits the autoencoder to the data.

        Parameters:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.
        '''
        self._model.fit(X, y, X_valid, y_valid)
    
    def get_latent(self, X):
        ''' 
        Gets the latent space representation of the data.
        '''
        latent = self._model.forward(X, encode=True)
        return latent
    
    def evaluate(self, X, y):
        ''' 
        Evaluates the model on the input data.
        '''
        return self._model.evaluate(X, y)

    
