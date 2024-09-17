import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
import pandas as pd
import sys
import os
from scipy.stats import multivariate_normal

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('gmm.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class GaussianMixtureModel():
    ''' 
    This class is used to perform Gaussian Mixture Model on the given dataset.
    '''
    def __init__(self) -> None:
        '''
        This function is used to initialize the class.
        '''
        self._u = None
        self._sigma = None
        self._pi = None
        self._membership = None
        self._likelihood = None
        self._data = None
        pass

    def getParams(self) -> tuple:
        ''' 
        This function is used to return the parameters of the Gaussian components in the 
        mixture model.
        '''
        return self._u, self._sigma, self._pi
    
    def getMembership(self, epsilon=1e-3) -> np.ndarray:
        ''' 
        This function is used to return the membership of the each sample in the dataset.

        Parameters:
            epsilon = float for preventing denominator to go to 0
        '''
        for k in range(self._K):
            gaussianPdf = multivariate_normal.pdf(self._data.to_numpy(), mean=self._u[k],
                                              cov=self._sigma[k] + np.eye(self._data.shape[1]) * epsilon, allow_singular=True)
            self._membership[:, k] = self._pi[k] * gaussianPdf
        sum_gamma = (np.sum(self._membership, axis=1) + epsilon)
        self._membership /= (sum_gamma[:, np.newaxis])
        return self._membership
    
    def getLikelihood(self, epsilon=1e-3):
        ''' 
        This function is used to return the overall likelihood of the entire dataset under 
        the current model parameters.

        Parameters:
            epsilon = float for preventing denominator to go to 0
        '''
        sum = np.sum(self._membership, axis=1) + epsilon
        self._likelihood = np.sum(np.log(sum))
        print("Log Likelihood: ", self._likelihood)
        return self._likelihood

    def InitParams(self, dataset: pd.DataFrame, K, init_method = "random_from_data"):
        ''' 
        This function is used to initialize the parameters of the model.

        Parameters:
            dataset = numpy array of shape (n_samples, n_features) containing the dataset.
            K = integer denoting the number of clusters.
            init_method = string denoting the method to initialize the parameters.
        '''
        self._data = dataset
        self._K = K
        n_samples, n_features = dataset.shape
        self._membership = np.zeros((n_samples, K))

        self._u = dataset.sample(K, random_state=42, replace=False).to_numpy()

        if init_method == 'random':
            self._sigma = np.random.randn(n_features, n_samples)
            self._sigma = np.dot(self._sigma, self._sigma.T)
            self._sigma = np.tile(self._sigma, (K, 1, 1))

        elif init_method == 'identity':
            self._sigma = np.array([np.eye(n_features) for i in range(K)])
        
        elif init_method == 'random_from_data':
            self._sigma = np.array([np.diag(np.var(dataset)) for i in range(K)])
            print(self._sigma.shape)

        self._pi = np.ones((K, 1))/K
        
        self._likelihood = 0
    
    def fit(self, dataset: pd.DataFrame, K, init_method = "random_from_data", epochs=100, epsilon=1e-3):
        ''' 
        This function is used to find the optimal model parameters by using the 
        Expectation-Maximization algorithm.

        Parameters:
            dataset = numpy array of shape (n_samples, n_features) containing the dataset.
            K = integer denoting the number of clusters.
            init_method = string denoting the method to initialize the parameters.
            epochs = integer denoting the number of iterations to be performed.
            epsilon = float for preventing denominator to go to 0
        '''
        self.InitParams(dataset, K, init_method)
        n_samples, n_features = self._data.shape

        for epoch in range(epochs):
            # E step
            print("Epoch: ", epoch, "Step: E")
            self._membership = self.getMembership(epsilon)

            # M step
            print("Epoch: ", epoch, "Step: M")
            sum_M = np.sum(self._membership, axis=0)
            for k in range(K):
                self._u[k] = np.sum(self._membership[:, k][:, np.newaxis] * self._data.to_numpy(), axis=0) / (sum_M[k] + epsilon)
                diff = self._data.to_numpy() - self._u[k]
                self._sigma[k] = np.dot((self._membership[:, k] * diff.T), diff) / (sum_M[k] + epsilon)
                self._pi[k] = sum_M[k] / n_samples

    def getCluster(self) -> np.ndarray:
        ''' 
        This function is used to return the cluster number of each sample in the dataset.
        '''
        return np.argmax(self._membership, axis=1)
    
    def Gaussian(self, x, u, sigma, epsilon=1e-3):
        ''' 
        This function is used to calculate the Gaussian probability of the given sample

        Parameters:
            u = numpy array of shape (n_features,) containing the mean of the Gaussian component.
            sigma = numpy array of shape (n_features, n_features) containing the covariance matrix of the Gaussian component.
            x = numpy array of shape (n_features,) containing the sample.
            epsilon = float for preventing denominator to go to 0
        '''
        n_features = x.shape[0]
        numerator = np.exp(-0.5*np.dot(np.dot((x-u).T, np.linalg.inv(sigma)), (x-u)))
        denominator = (2*np.pi)**(n_features/2)*np.sqrt(np.linalg.det(sigma))
        return numerator/(denominator + epsilon)
    
    def doAIC(self):
        ''' 
        This function is used to calculate the Akaike Information Criterion.
        '''
        return -2*self._likelihood + 2*(self._K**2 + 2*self._K)
    
    def doBIC(self):
        ''' 
        This function is used to calculate the Bayesian Information Criterion.
        '''
        return -2*self._likelihood + self._K*np.log(self._data.shape[0])
