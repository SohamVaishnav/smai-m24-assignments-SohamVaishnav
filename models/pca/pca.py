import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

class PCA():
    ''' 
    This class is used to perform PCA on the given dataset.
    '''
    def __init__(self, n_components=2) -> None:
        '''
        Initialize the PCA class.

        Parameters:
            n_components (int): The number of components to keep.
        '''
        self.n_components = n_components
        self._data_trans = None
        self._eigvals = None
        self._eigvecs = None
    
    def getComponents(self):
        ''' 
        Get the number of components being considered for PCA.
        '''
        return self.n_components
    
    def getEigenvalues(self):
        ''' 
        Get the eigenvalues of the covariance matrix.
        '''
        return self._eigvals

    def getEigenVectors(self):
        ''' 
        Get the eigenvectors of the covariance matrix.
        '''
        return self._eigvecs
    
    def fit(self, dataset: pd.DataFrame):
        ''' 
        Fit the PCA model to the dataset.

        Parameters:
            dataset (pd.DataFrame): The dataset on which PCA is to be performed
        '''
        self._mean = dataset.mean()
        self._data = dataset - dataset.mean()
        self.CovMat = self._data.cov()
        self._eigvals, self._eigvecs = np.linalg.eig(self.CovMat)
        self._eigvals = np.real(self._eigvals)
        self._eigvecs = np.real(self._eigvecs)
        sorted_idx = np.argsort(self._eigvals)[::-1]
        self._eigvals = self._eigvals[sorted_idx]
        self._eigvecs = self._eigvecs[:, sorted_idx]

        self._princComps = self._eigvecs[:, :self.n_components]

    def transform(self):
        ''' 
        Transform the dataset according to the fitted model.
        '''
        self._data_trans = self._data.dot(self._princComps)
        return self._data_trans

    def getExplainedVarRatio(self):
        ''' 
        Get the explained variance ratio of each principal component.
        '''
        return self._eigvals / np.sum(self._eigvals)

    def checkPCA(self) -> bool:
        ''' 
        Check whether the dimensions have been reduced correctly using the concept that maximising variance is 
        equivalent to minimising reconstruction error.
        '''
        cumsum = np.cumsum(self._eigvals)
        cumsum = cumsum/cumsum[-1]

        fig = plt.figure()
        plt.plot(range(1, len(cumsum)+1), cumsum, marker='o', linestyle='--')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs Number of Components')
        plt.grid()
        plt.show()

        return cumsum[self.n_components-1] >= 0.95
        # data_reconstructed = self._data_trans.dot(self._princComps.T)
        # return np.allclose(self._data, data_reconstructed, rtol=1e-5)