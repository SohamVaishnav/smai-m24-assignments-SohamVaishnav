import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.functional as F

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('pca_autoencoder.py')))
CurrDIR = os.path.dirname(os.path.abspath('pca_autoencoder.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/4/")

class PcaAutoencoder(nn.Module):
    '''
     PCA AutoEncoder class.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        '''
        Initialising the class.

        Parameters:
            input_dim: number of input features
            output_dim: number of output features  
        '''
        super(PcaAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.eigenvectors = None
        self.mean = None

    def fit(self, X):
        """
        Calculate the eigenvalues and eigenvectors of the covariance matrix.

        Parameters:
            X: (str) input data
        """
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        covariance_matrix = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1)

        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)

        self.eigenvalues = torch.real(eigenvalues)
        self.eigenvectors = torch.real(eigenvectors)

        idx = torch.argsort(self.eigenvalues, descending=True)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx][:, :self.output_dim]  # Shape: (input_dim, output_dim)

    def encode(self, X):
        """
        Reduce the dimensionality of the input data.

        Parameters
            X: (str) input data
        """
        X_centered = X - self.mean 
        return torch.mm(X_centered, self.eigenvectors)

    def decode(self, Z):
        """
        Reconstruct data from the latent space.

        Parameters
            Z: (str) latent space representation
        """
        return torch.mm(Z, self.eigenvectors.t()) + self.mean  # Shape: (batch_size, input_dim)

    def forward(self, X):
        """
        Forward pass.

        Parameters
            X: (str) input data
        """
        Z = self.encode(X)
        return self.decode(Z)

    def calculate_reconstruction_error(self, original, reconstructed):
        """
        Calculate the Mean Squared Error between original and reconstructed images.

        Parameters
            original: (str) original input data
            reconstructed: (str) reconstructed input
        """
        return torch.mean((original - reconstructed) ** 2)
