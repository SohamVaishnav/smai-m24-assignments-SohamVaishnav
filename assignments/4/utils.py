import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb

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
CurrDIR = os.path.dirname(os.path.abspath('utils.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/4/")

class MultiMNISTDataset(object):
    ''' 
    MultiMNISTDataset class.
    '''

    def __init__(self, root: str, batch_size: int):
        ''' 
        Parameters:
            root: str = path to the dataset.
            batch_size: int = batch size.
        '''
        self._root = root
        self._batch_size = batch_size

        assert os.path.exists(self._root), f"Path {self._root} does not exist."

    def load_mnist_data(self, task: str) -> None:
        '''
        Load the MNIST dataset.

        Parameters:
            task: str = the task to load the data for.
        '''
        self._data_path = os.path.join(self._root, 'double_mnist')
        self._train_path = os.path.join(self._data_path, 'train')
        self._valid_path = os.path.join(self._data_path, 'val')
        self._test_path = os.path.join(self._data_path, 'test')

        for path in [self._train_path, self._valid_path, self._test_path]:
            assert os.path.exists(path), f"Path {path} does not exist."

        self._train_images = []
        self._train_labels = []
        self._valid_images = []
        self._valid_labels = []
        self._test_images = []
        self._test_labels = []

        for root, _, files in os.walk(self._train_path):
            self._train_images.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._train_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._train_labels.extend([len(root[-1]) for file in files])
        
        for root, _, files in os.walk(self._valid_path):
            self._valid_images.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._valid_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._valid_labels.extend([len(root[-1]) for file in files])
        
        for root, _, files in os.walk(self._test_path):
            self._test_images.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._test_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._test_labels.extend([len(root[-1]) for file in files])
        
        pass
    
    def getLabels(self) -> None:
        '''
        Returns the labels for the dataset.
        '''
        labels = np.unique(self._train_labels + self._valid_labels + self._test_labels)
        self._labels
        pass
