import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
import pandas as pd
import sys
import os

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('pca.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class PCA():
    ''' 
    This class is used to perform PCA on the given dataset.
    '''
    def __init__(self, n_components: int) -> None:
        ''' 
        This function is used to initialize the class object.
        '''
        self.n_components = n_components
        pass

    def GetComponents(self):
        ''' 
        This function is used to get the number of components.
        '''
        return self.n_components
    
    def fit(self):
        ''' 
        This function is used to fit the model.
        '''
        pass

    def transform(self):
        ''' 
        This function is used to transform the model according to the fit.
        '''
        pass

    def checkPCA(self):
        ''' 
        This function is used to check whether the dimensions have been reduced.
        '''
        pass