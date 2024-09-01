import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
import pandas as pd
import sys
import os

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('gmm.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class GaussianMixtureModel():
    ''' 
    This class is used to perform Gaussian Mixture Model on the given dataset.
    '''
    def __init__(self) -> None:
        ''' 
        This function is used to initialize the class object.
        '''
        self._params = None
        self._membership = None
        self._likelihood = None
        pass

    def getParams(self):
        ''' 
        This function is used to return the parameters of the Gaussian components in the 
        mixture model.
        '''
        return self._params
    
    def getMembership(self):
        ''' 
        This function is used to return the membership of the each sample in the dataset.
        '''
        return self._membership
    
    def getLikelihood(self):
        ''' 
        This function is used to return the overall likelihood of the entire dataset under 
        the current model parameters.
        '''
        return self._likelihood
    
    def fit(self):
        ''' 
        This function is used to find the optimal model parameters by using the 
        Expectation-Maximization algorithm.
        '''
        pass
