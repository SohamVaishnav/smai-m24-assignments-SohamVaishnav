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

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('cnn.py')))
CurrDIR = os.path.dirname(os.path.abspath('cnn.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/4/")

from performance_measures.metricsMLP import *
from models.cnn.cnn import *

class Model_trainer:
    ''' 
    Trainer class for CNN model
    '''
    def __init__(self, config, ) -> None:
        ''' 
        Initialize the model

        Parameters:
            model: Model to be trained
            learning_rate: Learning rate for the optimizer
            loss: Loss function to be used
            optimizer: Optimizer to be used for training
        '''
        self._model = config['model']
        self._loss = config['loss']
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.to(self._device)
        self._lr = config['learning_rate']
        self._batch_size = config['batch_size']
        self._optimizer = config['optimizer']
        self._wb = config['wandb']
        self._labels = config['labels']
        self._labelsRnum = config['labelsRnum']
        self._FM_Vis = config['FM_Vis']
        pass 

    def accuracy(self, y_pred, y_true):
        if (self._model._task == 'classification'):
            _, preds = torch.max(y_pred, dim=1)
        elif (self._model._task == 'regression'):
            preds = torch.round(y_pred)
        return torch.tensor(torch.sum(preds == y_true).item() / len(preds))

    def trainer(self, X_train, y_train, X_valid, y_valid, epochs) -> None:
        ''' 
        Trains the model.

        Parameters:
            X_train: Training data
            y_train: Training labels
            X_valid: Validation data
            y_valid: Validation labels
            epochs: Number of epochs to train the model
        '''
        loss = 0
        train_acc = [] 
        val_acc = []
        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            self._optimizer.zero_grad()
            for batch in range(0, len(X_train), self._batch_size):
                img = np.array(X_train[batch:batch+self._batch_size])
                label = np.array(y_train[batch:batch+self._batch_size])

                img = torch.tensor(img).unsqueeze(1).to(self._device)
                if (self._model._task == 'classification'):
                    label = torch.tensor(label, dtype = torch.long).to(self._device)
                elif (self._model._task == 'regression'):
                    label = torch.tensor(label, dtype = torch.float).to(self._device)
                
                pred = self._model.forward(img)

                if (self._FM_Vis):
                    if (batch == self._batch_size or batch == self._batch_size*100 or batch == self._batch_size*150):
                        pred = self._model.forward(img, True)

                if (self._model._task == 'regression'):
                    pred = torch.reshape(pred, (-1,))

                if (self._loss == 'cross_entropy'):
                    loss = F.cross_entropy(pred, label)
                elif (self._loss == 'MSE'):
                    loss = F.mse_loss(pred, label)
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()

            acc, loss = self.evaluate(X_train, y_train, False)
            train_acc.append(np.mean(acc))
            train_loss.append(np.mean(loss))
            acc, loss = self.evaluate(X_valid, y_valid, False)
            val_acc.append(np.mean(acc))
            val_loss.append(np.mean(loss))

            # if (self._wb):
                # wandb.log({'Train Loss': np.mean(loss_train), 'Valid Loss': np.mean(loss_valid)})
                # wandb.log({'Train Accuracy': acc_train, 'Valid Accuracy': acc_valid})
                # wandb.log({'Train Precision': prec_train[0], 'Valid Precision': prec_valid[0]})
                # wandb.log({'Train Recall': rec_train[0], 'Valid Recall': rec_valid[0]})
                # wandb.log({'Train F1 Score': f1_train[0], 'Valid F1 Score': f1_valid[0]})
                # wandb.log({'epoch': epoch})
        self.plot_metrics(train_acc, val_acc, train_loss, val_loss)     
        pass

    def evaluate(self, X_test, y_test, isTest: bool) -> None:
        '''
        Evaluate the model.

        Parameters:
            X_test: Testing data
            y_test: Testing labels
            isTest: Whether the data is test data or not
        '''
        loss = []
        acc = []
        y_pred = []
        with torch.no_grad():
            for batch in range(0, len(X_test), self._batch_size):
                img = np.array(X_test[batch:batch+self._batch_size])
                label = np.array(y_test[batch:batch+self._batch_size])
                
                img = torch.tensor(img).unsqueeze(1).to(self._device)
                if (self._model._task == 'classification'):
                    label = torch.tensor(label, dtype = torch.long).to(self._device)
                elif (self._model._task == 'regression'):
                    label = torch.tensor(label, dtype = torch.float).to(self._device)
                y_pred.append(self._model.forward(img))

                if (self._model._task == 'regression'):
                    y_pred[-1] = torch.reshape(y_pred[-1], (-1,))
                
                if (self._loss == 'cross_entropy'):
                    loss.append(F.cross_entropy(y_pred[-1], label))
                elif (self._loss == 'MSE'):
                    loss.append(F.mse_loss(y_pred[-1], label))
                
                acc.append(self.accuracy(y_pred[-1], label))
            
            print("Acc = ", np.mean(acc))
            print("Loss = ", np.mean(loss))
        return acc, loss

    def plot_metrics(self, train_acc, val_acc, train_loss, val_loss):
        plt.figure(figsize=(10, 5))
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

class CNN(nn.Module):
    '''
    CNN model 
    '''

    def __init__(self, config):
        ''' 
        Initialize the model

        Parameters:
            config: Configuration for the model
        '''
        super(CNN, self).__init__()
        self._task = config['task']
        self._in_channels = config['in_channels']
        self._ConvLayers = config['ConvLayers']
        self._FCLayers = config['FCLayers']
        self._pool = config['pool']
        self._kernel_size = config['kernel_size']
        self._strides = config['strides']
        self._dropout = config['dropout']
        self._activation = config['activation']
        self._num_classes = self._FCLayers[-1]

        self._convs = nn.ModuleList()
        for i in range(len(self._ConvLayers)):
            if (i == 0):
                self._convs.append(nn.Conv2d(in_channels=self._in_channels, out_channels=self._ConvLayers[i], 
                                             kernel_size=self._kernel_size[i], padding = self._kernel_size[i]//2))
            else:
                self._convs.append(nn.Conv2d(in_channels=self._ConvLayers[i-1], out_channels=self._ConvLayers[i], 
                                             kernel_size=self._kernel_size[i], padding = self._kernel_size[i]//2))
        
        self._fcs = nn.ModuleList()
        for i in range(len(self._FCLayers)):
            if (i != len(self._FCLayers)-1):
                self._fcs.append(nn.Linear(in_features=self._FCLayers[i], out_features=self._FCLayers[i+1]))
            if (self._dropout is not None):
                self._fcs.append(nn.Dropout(self._dropout))
        pass

    def forward(self, input, Viz = False):
        '''
        Forward pass for the model.

        Parameters:
            input: Input to the model
            Viz: Whether to visualize the feature maps or not
        '''
        if (Viz):
            fig = plt.figure(figsize=(10, 5))
            fig.suptitle('Feature Maps', fontsize=16)
            plt.tight_layout()
            j = 1
        for i in range(len(self._ConvLayers)):
            if isinstance(self._convs[i], nn.Conv2d):
                input = self._convs[i](input)
            elif isinstance(self._convs[i], nn.MaxPool2d):
                input = self._convs[i](input)
            if (self._activation == 'relu' and i != len(self._ConvLayers)-1):
                input = F.relu(input)
            elif (self._activation == 'tanh' and i != len(self._ConvLayers)-1):
                input = F.tanh(input)
            elif (self._activation == 'sigmoid' and i != len(self._ConvLayers)-1):
                input = F.sigmoid(input)
            elif (self._activation == 'softmax' and i != len(self._ConvLayers)-1):
                input = F.softmax(input)
            if (Viz):
                ax = plt.subplot(1, 3, j)
                ax.imshow(input[0, 0].detach().cpu().numpy())
                ax.set_title('Layer #{}'.format(j))
                ax.axis('off')
                j += 1
            if (self._pool is not None):
                input = F.max_pool2d(input, self._pool[i])
        if (Viz):
            plt.show()
        
        input = input.view(input.size(0), -1)
        for i in range(len(self._FCLayers)):
            input = self._fcs[i](input)
            if (self._activation == 'relu' and i != len(self._FCLayers)-1):
                input = F.relu(input)
            elif (self._activation == 'tanh' and i != len(self._FCLayers)-1):
                input = F.tanh(input)
            elif (self._activation == 'sigmoid' and i != len(self._FCLayers)-1):
                input = F.sigmoid(input)
            # elif (i == len(self._FCLayers)-1):
            #     input = F.softmax(input)
            # if (self._dropout is not None and i != len(self._FCLayers)-2):
            #     input = F.dropout(input, self._dropout)
        
        return input
