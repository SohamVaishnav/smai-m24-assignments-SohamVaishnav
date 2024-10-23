import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb
import PIL

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

    def __init__(self, batch_size: int, root: str = RawDataDIR):
        ''' 
        Parameters:
            root: str = path to the dataset.
            batch_size: int = batch size.
        '''
        self._root = root
        self._batch_size = batch_size
        self._resizer = transforms.Compose([transforms.ToPILImage(), 
                                        transforms.Resize((28, 28)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

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

        self._train_images_paths = []
        self._train_images = []
        self._train_labels = []

        self._valid_images_paths = []
        self._valid_images = []
        self._valid_labels = []

        self._test_images_paths = []
        self._test_images = []
        self._test_labels = []

        for root, _, files in os.walk(self._train_path):
            self._train_images_paths.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._train_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._train_labels.extend([0 for file in files])
        
        for i in range(len(self._train_images_paths)):
            self._train_images.append(cv2.imread(self._train_images_paths[i], cv2.IMREAD_GRAYSCALE))
            # self._train_images[-1] = self._resizer(self._train_images[-1])
            # self._train_images[-1] = self._train_images[-1].numpy()
            # self._train_images[-1] = np.transpose(self._train_images[-1], (1, 2, 0))
            self._train_images[-1] = cv2.resize(self._train_images[-1], (28, 28))
            self._train_images[-1] = self._train_images[-1].astype(np.float32)
            self._train_images[-1] /= 255.0
        
        for root, _, files in os.walk(self._valid_path):
            self._valid_images_paths.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._valid_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._valid_labels.extend([0 for file in files])
        
        for i in range(len(self._valid_images_paths)):
            self._valid_images.append(cv2.imread(self._valid_images_paths[i], cv2.IMREAD_GRAYSCALE))
            # self._valid_images[-1] = self._resizer(self._valid_images[-1])
            # self._valid_images[-1] = self._valid_images[-1].numpy()
            # self._valid_images[-1] = np.transpose(self._valid_images[-1], (1, 2, 0))
            self._valid_images[-1] = cv2.resize(self._valid_images[-1], (28, 28))
            self._valid_images[-1] = self._valid_images[-1].astype(np.float32)
            self._valid_images[-1] /= 255.0
        
        for root, _, files in os.walk(self._test_path):
            self._test_images_paths.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._test_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._test_labels.extend([0 for file in files])
        
        for i in range(len(self._test_images_paths)):
            self._test_images.append(cv2.imread(self._test_images_paths[i], cv2.IMREAD_GRAYSCALE))
            # self._test_images[-1] = self._resizer(self._test_images[-1])
            # self._test_images[-1] = self._test_images[-1].numpy()
            # self._test_images[-1] = np.transpose(self._test_images[-1], (1, 2, 0))
            self._test_images[-1] = cv2.resize(self._test_images[-1], (28, 28))
            self._test_images[-1] = self._test_images[-1].astype(np.float32)
            self._test_images[-1] /= 255.0
        
        pass
    
    def getLabels(self) -> None:
        '''
        Returns the labels for the dataset.
        '''
        labels = np.unique(self._train_labels + self._valid_labels + self._test_labels)
        self._labels = labels
        return self._labels