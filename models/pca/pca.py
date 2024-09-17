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
    def __init__(self, n_components = 2) -> None:
        '''
        This function is used to initialize the class.

        Parameters:
            n_components (int): The number of components to keep.
        '''
        self.n_components = n_components
        pass

    def getComponents(self):
        ''' 
        This function is used to get the number of components being considered for PCA.
        '''
        return self.n_components
    
    def getEigenvalues(self):
        ''' 
        This function is used to get the eigenvalues of the covariance matrix.
        '''
        return self._eigvals

    def getEigenVectors(self):
        ''' 
        This function is used to get the eigenvectors of the covariance matrix.
        '''
        return self._eigvecs
    
    def fit(self, dataset: pd.DataFrame):
        ''' 
        This function is used to fit the model.

        Parameters:
            dataset (pd.DataFrame): The dataset on which PCA is to be performed
        '''
        self._data = dataset

        self.CovMat = dataset.cov()
        self._eigvals, self._eigvecs = np.linalg.eig(self.CovMat)
        self._eigvals = np.sort(np.real(self._eigvals))[::-1]
        self._eigvecs = np.real(self._eigvecs)[self._eigvals.argsort()[::-1]]

        self._princComps = self._eigvecs[:self.n_components]
        return None

    def transform(self):
        ''' 
        This function is used to transform the model according to the fit.
        '''
        data_trans = self._data.dot(self._princComps.T)
        self._data_trans = data_trans
        return data_trans

    def checkPCA(self) -> bool:
        ''' 
        This function is used to check whether the dimensions have been reduced to the correct eigenvectors.
        '''
        data_reconstructed = self._data_trans.dot(self._princComps) + self._data.mean()
        return np.allclose(data_reconstructed, self._data, atol=1e-5, rtol=1e-5)