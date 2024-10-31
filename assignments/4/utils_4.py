import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb
import struct

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

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
                                        transforms.Resize((64, 64)), 
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
            elif (task == 'multi_label'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._train_labels.extend([int(file.split('_')[0])+1 for file in files])
                else:
                    self._train_labels.extend([-1 for file in files])
        
        for i in range(len(self._train_images_paths)):
            self._train_images.append(cv2.imread(self._train_images_paths[i], cv2.IMREAD_GRAYSCALE))
            self._train_images[-1] = self._resizer(self._train_images[-1])
            self._train_images[-1] = self._train_images[-1].numpy()
            # self._train_images[-1] = np.transpose(self._train_images[-1], (1, 2, 0))
            # self._train_images[-1] = cv2.resize(self._train_images[-1], (64, 64))
            # self._train_images[-1] = self._train_images[-1].astype(np.float32)
            # self._train_images[-1] /= 255.0
        
        for root, _, files in os.walk(self._valid_path):
            self._valid_images_paths.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._valid_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._valid_labels.extend([0 for file in files])
            elif (task == 'multi_label'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._valid_labels.extend([int(file.split('_')[0])+1 for file in files])
                else:
                    self._valid_labels.extend([-1 for file in files])
        
        for i in range(len(self._valid_images_paths)):
            self._valid_images.append(cv2.imread(self._valid_images_paths[i], cv2.IMREAD_GRAYSCALE))
            self._valid_images[-1] = self._resizer(self._valid_images[-1])
            self._valid_images[-1] = self._valid_images[-1].numpy()
            # self._valid_images[-1] = np.transpose(self._valid_images[-1], (1, 2, 0))
            # self._valid_images[-1] = cv2.resize(self._valid_images[-1], (64, 64))
            # self._valid_images[-1] = self._valid_images[-1].astype(np.float32)
            # self._valid_images[-1] /= 255.0
        
        for root, _, files in os.walk(self._test_path):
            self._test_images_paths.extend([os.path.join(root, file) for file in files])
            if (task == 'num_digits'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._test_labels.extend([len(file.split('_')[-1].removesuffix('.png')) for file in files])
                else:
                    self._test_labels.extend([0 for file in files])
            elif (task == 'multi_label'):
                if ([len(file.split('_')) > 1 for file in files] == [True for file in files]):
                    self._test_labels.extend([int(file.split('_')[0]) for file in files])
                else:
                    self._test_labels.extend([-1 for file in files])
        
        for i in range(len(self._test_images_paths)):
            self._test_images.append(cv2.imread(self._test_images_paths[i], cv2.IMREAD_GRAYSCALE))
            self._test_images[-1] = self._resizer(self._test_images[-1])
            self._test_images[-1] = self._test_images[-1].numpy()
            # self._test_images[-1] = np.transpose(self._test_images[-1], (1, 2, 0))
            # self._test_images[-1] = cv2.resize(self._test_images[-1], (64, 64))
            # self._test_images[-1] = self._test_images[-1].astype(np.float32)
            # self._test_images[-1] /= 255.0
        
        pass
    
    def getLabels(self, task: str) -> None:
        '''
        Returns the labels for the dataset.

        Parameters:
            task: str = the task to get the labels for.
        '''
        if (task == 'num_digits'):
            self._labels = np.unique(self._train_labels + self._valid_labels + self._test_labels)
        elif (task == 'multi_label'):
            labels = np.unique(self._train_labels + self._valid_labels + self._test_labels)
            self._labels = self.OneHotEncoder(labels)
        return self._labels

    def OneHotEncoder(self, curr_labels: list) -> np.ndarray:
        ''' 
        One hot encodes the labels.

        Parameters:
            curr_labels: list = the labels to one hot encode.
        '''
        labels = np.zeros((len(curr_labels), 33))
        for i in range(len(curr_labels)):
            temp = str(curr_labels[i])
            if (len(temp) == 1):
                labels[i][int(temp)] = 1
                labels[i][21] = 1
                labels[i][32] = 1
            elif (len(temp) == 2 and temp != '-1'):
                labels[i][int(temp[0])] = 1
                labels[i][11+int(temp[1])] = 1
                labels[i][32] = 1
            elif (len(temp) == 3):
                labels[i][int(temp[0])] = 1
                labels[i][11+int(temp[1])] = 1
                labels[i][22+int(temp[2])] = 1
            if (temp == '-1'):
                labels[i][10] = 1
                labels[i][21] = 1
                labels[i][32] = 1
        return labels
    
# DL = MultiMNISTDataset(32, RawDataDIR)
# DL.load_mnist_data('multi_label')
# labels = DL.getLabels('multi_label')
# labs = DL.OneHotEncoder(DL._test_labels)

# df = pd.DataFrame(np.column_stack((labs, np.array(DL._test_labels)-1)), columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
#                                                                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
#                                                                             '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', 'digits'])
# df.to_csv(os.path.join(PreProcessDIR, 'multi_label_test_labels.csv'), index=False)

# class FashionMNISTDataset(Dataset):
#     '''
#     A custom Dataset class to handle FashionMNIST binary data (.ubyte files).
#     '''
#     def __init__(self, images_path: str, labels_path: str, transform=None):
#         self.images = self._read_images(images_path)
#         self.labels = self._read_labels(labels_path)
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, label

#     def _read_images(self, file_path: str):
#         '''Read images from the binary .ubyte file.'''
#         with open(file_path, 'rb') as f:
#             magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
#             images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
#             images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]
#         return images

#     def _read_labels(self, file_path: str):
#         '''Read labels from the binary .ubyte file.'''
#         with open(file_path, 'rb') as f:
#             magic, num = struct.unpack(">II", f.read(8))
#             labels = np.frombuffer(f.read(), dtype=np.uint8)
#         return labels

# class FashionMNISTLoader:
#     '''
#     A class to load FashionMNIST dataset from binary .ubyte files.
#     '''
#     def __init__(self, batch_size: int, root: str = RawDataDIR):
#         '''
#         Parameters:
#             root (str): Path to the dataset directory.
#             batch_size (int): Batch size for data loaders.
#         '''
#         self._root = os.path.join(root, 'fashionMNIST')
#         self._batch_size = batch_size
#         self._transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((28, 28)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])

#         assert os.path.exists(self._root), f"Path {self._root} does not exist."

#     def _get_data_paths(self, dataset_type: str):
#         '''
#         Get the paths to the images and labels .ubyte files.
#         '''
#         images_path = os.path.join(self._root, f'{dataset_type}-images-idx3-ubyte')
#         labels_path = os.path.join(self._root, f'{dataset_type}-labels-idx1-ubyte')
#         assert os.path.exists(images_path), f"{images_path} does not exist."
#         assert os.path.exists(labels_path), f"{labels_path} does not exist."
#         return images_path, labels_path

#     def _create_dataloader(self, dataset_type: str):
#         '''
#         Create a DataLoader for the given dataset type (train/val/test).
#         '''
#         images_path, labels_path = self._get_data_paths(dataset_type)
#         dataset = FashionMNISTDataset(images_path, labels_path, transform=self._transform)
#         return DataLoader(dataset, batch_size=self._batch_size, shuffle=(dataset_type == 'train'))

#     def load_data(self):
#         '''
#         Load the train, validation, and test DataLoaders.
#         '''
#         train_loader = self._create_dataloader('train')
#         test_loader = self._create_dataloader('t10k')
#         return train_loader, test_loader


import os
import struct
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch

class FashionMNISTDataset(Dataset):
    '''
    A custom Dataset class to handle FashionMNIST binary data (.ubyte files).
    '''
    def __init__(self, images_path: str, labels_path: str, transform=None):
        self.images = self._read_images(images_path)
        self.labels = self._read_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')

        if self.transform:
            image = self.transform(image)

        return image, label

    def _read_images(self, file_path: str):
        '''Read images from the binary .ubyte file.'''
        with open(file_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            images = images.astype(np.float32) / 255.0 
        return images

    def _read_labels(self, file_path: str):
        '''Read labels from the binary .ubyte file.'''
        with open(file_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


class FashionMNISTLoader:
    '''
    A class to load FashionMNIST dataset from binary .ubyte files.
    '''
    def __init__(self, batch_size: int, root: str = RawDataDIR):
        '''
        Parameters:
            root (str): Path to the dataset directory.
            batch_size (int): Batch size for data loaders.
        '''
        self._root = os.path.join(root, 'fashionMNIST')
        self._batch_size = batch_size
        self._transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        assert os.path.exists(self._root), f"Path {self._root} does not exist."

    def _get_data_paths(self, dataset_type: str):
        '''
        Get the paths to the images and labels .ubyte files.
        '''
        images_path = os.path.join(self._root, f'{dataset_type}-images-idx3-ubyte')
        labels_path = os.path.join(self._root, f'{dataset_type}-labels-idx1-ubyte')
        assert os.path.exists(images_path), f"{images_path} does not exist."
        assert os.path.exists(labels_path), f"{labels_path} does not exist."
        return images_path, labels_path

    def _create_dataloader(self, dataset_type: str, shuffle: bool = False):
        '''
        Create a DataLoader for the given dataset type (train/val/test).
        '''
        images_path, labels_path = self._get_data_paths(dataset_type)
        dataset = FashionMNISTDataset(images_path, labels_path, transform=self._transform)
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle)

    def load_data(self):
        '''
        Load the train, validation, and test DataLoaders.
        Splits train dataset into 80% train and 20% validation.
        '''
        images_path, labels_path = self._get_data_paths('train')
        full_train_dataset = FashionMNISTDataset(images_path, labels_path, transform=self._transform)

        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        test_loader = self._create_dataloader('t10k', shuffle=False)

        return train_loader, val_loader, test_loader
