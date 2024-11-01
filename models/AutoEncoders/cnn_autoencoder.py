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

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('cnn_autoencoder.py')))
CurrDIR = os.path.dirname(os.path.abspath('cnn_autoencoder.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/4/")

from models.cnn import cnn
from models.cnn import multilabel_cnn
from models.AutoEncoders import auto_encoder, cnn_autoencoder, pca_autoencoder

class CnnAutoencoder(nn.Module):
    '''
    CNN AutoEncoder class.
    '''
    def __init__(self, config):
        '''
        Initialising the class.

        Parameters:
            config: dictionary containing the configuration parameters 
        '''
        super(CnnAutoencoder, self).__init__()
        self.config = config
        self.in_channels = config['in_channels']
        self.encoder_layers = config['encoder_layers']
        self.decoder_layers = config['decoder_layers']
        self.lr = config['lr']

        encoder_layers = []
        length = len(self.encoder_layers)
        i = 0
        in_channels = self.in_channels
        for out_channels, kernel_size, stride, padding in self.encoder_layers:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if (i != length - 1):
              if (self.activation == 'relu'):
                encoder_layers.append(nn.ReLU(True))
            elif (self.activation == 'sigmoid'):
                encoder_layers.append(nn.Sigmoid())
            elif (self.activation == 'tanh'):
                encoder_layers.append(nn.Tanh())
            in_channels = out_channels
            i += 1
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        length = len(self.decoder_layers)
        i = 0
        for out_channels, kernel_size, stride, padding, output_padding in self.decoder_layers:
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
            if (i != length - 1):
                if (self.activation == 'relu'):
                    decoder_layers.append(nn.ReLU(True))
                elif (self.activation == 'sigmoid'):
                    decoder_layers.append(nn.Sigmoid())
                elif (self.activation == 'tanh'):
                    decoder_layers.append(nn.Tanh())
            in_channels = out_channels
            i += 1
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        '''
        Encode the input data.

        Parameters:
            x: input data 
        '''
        return self.encoder(x)

    def decode(self, z):
        '''
        Decode the latent space representation.

        Parameters:
            z: latent space representation
        '''
        return self.decoder(z)

    def forward(self, x):
        '''
        Forward pass of the network.

        Parameters:
            x: input data 
        '''
        latent = self.encode(x)
        return self.decode(latent)
