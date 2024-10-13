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

    def load_data(self):
        pass

