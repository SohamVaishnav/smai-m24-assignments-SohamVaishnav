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

from models.cnn import cnn
from models.cnn import multilabel_cnn

import torch

# class PcaAutoencoder(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int):
#         super(PcaAutoencoder, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.eigenvectors = None
#         self.eigenvalues = None

#     def fit(self, X):
#         """
#         Calculate the eigenvalues and eigenvectors of the covariance matrix.

#         Parameters
#             X : torch.Tensor
#                 Input data.
#         """
#         # Center the data
#         X_centered = X - torch.mean(X, dim=0)
#         covariance_matrix = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1)

#         # Calculate eigenvalues and eigenvectors
#         eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)

#         # Only take the real part of the eigenvalues and eigenvectors
#         self.eigenvalues = torch.real(eigenvalues)
#         self.eigenvectors = torch.real(eigenvectors)

#         # Sort eigenvalues and corresponding eigenvectors
#         idx = torch.argsort(self.eigenvalues, descending=True)
#         self.eigenvalues = self.eigenvalues[idx]
#         self.eigenvectors = self.eigenvectors[:, idx][:, :self.output_dim]

#     def encode(self, X):
#         """
#         Reduce the dimensionality of the input data.

#         Parameters
#             X : torch.Tensor
#                 Input data.
#         """
#         X_centered = X - torch.mean(X, dim=0)
#         return torch.mm(X_centered, self.eigenvectors)

#     def decode(self, Z):
#         """
#         Reconstruct data from the latent space.

#         Parameters
#             Z : torch.Tensor
#                 Latent space representation.
#         """
#         print("Z: ", Z.shape)
#         print("eig: ", torch.mean(Z, dim=0).shape)
#         return torch.mm(Z, self.eigenvectors.t()) + torch.mean(self.eigenvectors, dim=0)

#     def forward(self, X):
#         """
#         Forward pass.

#         Parameters
#             X : torch.Tensor
#                 Input data.
#         """
#         Z = self.encode(X)
#         return self.decode(Z)

#     def calculate_reconstruction_error(self, original, reconstructed):
#         """
#         Calculate the Mean Squared Error between original and reconstructed images.

#         Parameters
#             original : torch.Tensor
#                 Original input data.
#             reconstructed : torch.Tensor
#                 Reconstructed input
#         """
#         return torch.mean((original - reconstructed) ** 2)

# def plot_elbow_curve(errors, components):
#     """
#     Plot the Elbow curve for the PCA reconstruction error.

#     Parameters
#         components : list
#     """
#     plt.figure(figsize=(10, 5))
#     plt.plot(components, errors, marker='o')
#     plt.title('Elbow Curve: Reconstruction Error vs. Number of Components')
#     plt.xlabel('Number of Components')
#     plt.ylabel('Reconstruction Error (MSE)')
#     plt.grid()
#     plt.show()


import torch
import torch.nn as nn

class PcaAutoencoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(PcaAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.eigenvectors = None
        self.mean = None

    def fit(self, X):
        """
        Calculate the eigenvalues and eigenvectors of the covariance matrix.
        """
        # Center the data and calculate the mean
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        covariance_matrix = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)

        # Only take the real part of the eigenvalues and eigenvectors
        self.eigenvalues = torch.real(eigenvalues)
        self.eigenvectors = torch.real(eigenvectors)

        # Sort eigenvalues and corresponding eigenvectors
        idx = torch.argsort(self.eigenvalues, descending=True)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx][:, :self.output_dim]  # Shape: (input_dim, output_dim)

    def encode(self, X):
        """
        Reduce the dimensionality of the input data.
        """
        X_centered = X - self.mean  # Use the mean computed during fit
        return torch.mm(X_centered, self.eigenvectors)  # Shape: (batch_size, output_dim)

    def decode(self, Z):
        """
        Reconstruct data from the latent space.
        """
        return torch.mm(Z, self.eigenvectors.t()) + self.mean  # Shape: (batch_size, input_dim)

    def forward(self, X):
        """
        Forward pass.
        """
        Z = self.encode(X)
        return self.decode(Z)

    def calculate_reconstruction_error(self, original, reconstructed):
        """
        Calculate the Mean Squared Error between original and reconstructed images.
        """
        return torch.mean((original - reconstructed) ** 2)
