import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
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
        self._sum_gamma = np.sum(self._membership, axis=1)
        self._membership /= (self._sum_gamma[:, np.newaxis] + epsilon)
        return self._membership
    
    def getLikelihood(self):
        ''' 
        This function is used to return the overall likelihood of the entire dataset under 
        the current model parameters.

        Parameters:
            epsilon = float for preventing denominator to go to 0
        '''
        self._likelihood = np.sum(np.log(self._sum_gamma))
        print("Log Likelihood: ", self._likelihood)
        return self._likelihood

    def InitParams(self, dataset: pd.DataFrame, K, epsilon, init_method = "random_from_data"):
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

        self._u = dataset.sample(K, random_state=42).to_numpy()

        if init_method == 'random':
            self._sigma = np.random.randn(n_features, n_samples)
            self._sigma = np.dot(self._sigma, self._sigma.T)
            self._sigma = np.tile(self._sigma, (K, 1, 1))
            self._pi = np.random.rand(K, 1)

        elif init_method == 'identity':
            self._sigma = np.array([np.eye(n_features) for i in range(K)])
            self._pi = np.ones((K, 1))/K

        elif init_method == 'random_from_data':
            deviation = dataset - dataset.mean()
            cov = np.dot(deviation.T, deviation)
            self._sigma = np.array([np.cov(dataset.T) + np.eye(n_features)*epsilon for i in range(K)])
            print(self._sigma.max(), self._sigma.min())
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
        self.InitParams(dataset, K, epsilon, init_method)
        n_samples, n_features = self._data.shape

        for epoch in range(epochs):
            # E step
            print("Epoch: ", epoch, "Step: E")
            self._membership = self.getMembership(epsilon)

            # M step
            print("Epoch: ", epoch, "Step: M")
            sum_M = np.sum(self._membership, axis=0)
            for k in range(K):
                self._u[k] = np.sum(self._membership[:, k][:, np.newaxis] * self._data.to_numpy(), axis=0) / (sum_M[k])
                deviation = self._data.to_numpy() - self._u[k]
                self._sigma[k] = np.dot((self._membership[:, k] * deviation.T), deviation) / (sum_M[k])
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
        return -2*self.getLikelihood() + 2*self._K 
        # cov_params = self._K * self._data.shape[1] * (self._data.shape[1] + 1) / 2.0
        # mean_params = self._K * self._data.shape[1]
        # K = cov_params + mean_params + self._K - 1
        # return -2*self.getLikelihood() + 2*K
    
    def doBIC(self):
        ''' 
        This function is used to calculate the Bayesian Information Criterion.
        '''
        return -2*self._likelihood + self._K*np.log(self._data.shape[0])
    
    def plotGaussians(self):
        ''' 
        This function is used to plot the Gaussian components of the mixture model.
        '''
        #plot 2D gaussian ellipses
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.scatter(self._data.iloc[:, 0], self._data.iloc[:, 1], c=self.getCluster(), s=50, cmap='viridis')
        for k in range(self._K):
            mean = self._u[k]
            cov = self._sigma[k]
            v, w = np.linalg.eigh(cov)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi
            ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], color='r', alpha=0.5)
            ax.add_patch(ell)
        plt.show()

