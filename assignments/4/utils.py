import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb
import zipfile

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('utils.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class MultiMNISTDataset(object):
    ''' 
    MultiMNISTDataset class.
    '''

    def __init__(self, root: str, train: bool, valid: bool, batch_size: int):
        ''' 
        Parameters:
            root: str = path to the dataset.
            train: bool = whether to load the training data.
            valid: bool = whether to load the validation data.
            batch_size: int = batch size.
        '''
        self.root = root
        self.train = train
        self.valid = valid
        self.batch_size = batch_size

        assert os.path.exists(self.root), f"Path {self.root} does not exist."

    def load_mnist_data(self) -> None:
        '''
        Load the MNIST dataset.
        '''
        self._data_path = os.path.join(self.root, 'double_mnist')
        self._train_path = os.path.join(self._data_path, 'train')
        self._valid_path = os.path.join(self._data_path, 'valid')
        self._test_path = os.path.join(self._data_path, 'test')

        for path in [self._train_path, self._valid_path, self._test_path]:
            assert os.path.exists(path), f"Path {path} does not exist."

        self._train_images = []
        self._valid_images = []
        self._test_images = []

        for root, _, files in os.walk(self._train_path):
            self._train_images.extend([os.path.join(root, file) for file in files])
        for root, _, files in os.walk(self._valid_path):
            self._valid_images.extend([os.path.join(root, file) for file in files])
        for root, _, files in os.walk(self._test_path):
            self._test_images.extend([os.path.join(root, file) for file in files])
        
        pass

DL = MultiMNISTDataset()
