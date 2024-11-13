import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from scipy.stats import multivariate_normal

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('kde.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class KDE(object):
    '''
    Kernel Density Estimation 
    '''
    def __init__(self, data):
        self.data = data
        self.d = len(data[0])
        self.n = len(data)
        self.kde = None
    
    def buildKernel(self, type: str):
        '''
        Build the kernel function
        '''
        if (type == 'box'):
            self.kernel = self.boxKernel
        elif (type == 'gaussian'):
            self.kernel = self.gaussianKernel
        elif (type == 'triangular'):
            self.kernel = self.triangularKernel

    def boxKernel(self, x, thresh: float = 0.5):
        '''
        Box kernel

        Parameters:
            x: (float) the value to be evaluated
            thresh: (float) the threshold value
        '''
        self.thresh = thresh
        if (abs(x) <= thresh):
            return 1
        else:
            return 0
    
    def gaussianKernel(self, x, thresh: float = 0.5):
        '''
        Gaussian kernel

        Parameters:
            x: (float) the value to be evaluated
            thresh: (float) the threshold value
        '''
        self.thresh = thresh
        return np.exp(-(x**2)/(2*thresh**2)) / np.sqrt(2*np.pi*thresh**2)
    
    def triangularKernel(self, x, thresh: float = 0.5):
        '''
        Triangular kernel

        Parameters:
            x: (float) the value to be evaluated
            thresh: (float) the threshold value
        '''
        self.thresh = thresh
        if (abs(x) <= thresh):
            return 1 - abs(x/thresh)
        else:
            return 0
        
    def fit(self, kernel: str, params: dict):
        '''
        Fit the KDE model
        '''
        self.buildKernel(kernel, params)
        self.kde = np.zeros(self.n)
        distances = np.linalg.norm(self.data[:, np.newaxis] - self.data, axis=2)
        kernel_values = self.kernel(distances, params['thresh'])
        self.kde = np.mean(kernel_values, axis=1)
    
    def predict(self, x):
        '''
        Predict the density of the given point
        '''
        distances = np.linalg.norm(x - self.data, axis=1)
        kernel_values = self.kernel(distances, self.thresh)
        return np.mean(kernel_values)
    
    def plot(self, data, title: str):
        '''
        Plot the KDE model
        '''
        fig = px.scatter_3d(x=data[:, 0], y=data[:, 1], title=title)
        fig.update_traces(marker=dict(size=3))
        fig.show()
    

