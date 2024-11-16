import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('kde.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

class BitCounterRNN(nn.Module):
    '''
    Bit Counter
    '''
    def __init__(self, config):
        super(BitCounterRNN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.rnn = nn.RNN(config['input_size'], self.hidden_size, self.n_layers, 
                          batch_first=True, dropout=config['dropout'], nonlinearity=config['nonlinearity'])
        if (config['batch_norm']):
            self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        '''
        Forward pass

        Parameters:
            x: (tensor) the input tensor
        '''
        # h0 = torch.zeros(self.n_layers, x.size(), self.hidden_size)
        out, _ = self.rnn(x)
        if hasattr(self, 'batch_norm'):
            out = self.batch_norm(out[:, -1, :])
        else:
            out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    def predict(self, x):
        '''
        Predict the output

        Parameters:
            x: (tensor) the input tensor
        '''
        return self.forward(x)

    def fit(self, train_loader, val_loader, n_epochs, lr, 
            criterion = None, optimizer = None):
        '''
        Fit the model

        Parameters:
            x_train: (tensor) the input tensor for training
            y_train: (tensor) the target tensor for training
            x_val: (tensor) the input tensor for validation
            y_val: (tensor) the target tensor for validation
            n_epochs: (int) the number of epochs
            lr: (float) the learning rate
            batch_size: (int) the batch size
        '''
        if (criterion is None):
            criterion = nn.MSELoss()
        if (optimizer is None):
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(x_batch)
                outputs = outputs.squeeze(1)
                # train_MAE = F.l1_loss(outputs, y_batch)
                # loss = criterion(outputs, y_batch)
                loss = F.l1_loss(outputs, y_batch)
                loss.backward()
                optimizer.step()

            val_loss = self.evaluate(val_loader, criterion)

            print(f'Epoch {epoch+1}/{n_epochs}, MAE Train Loss: {loss.item()}, Val Loss: {val_loss}')
        print("Model trained successfully")

    def evaluate(self, val_loader, loss_fn = None):
        '''
        Evaluate the model

        Parameters:
            x: (tensor) the input tensor
            y: (tensor) the target tensor
        '''
        if (loss_fn is None):
            loss_fn = F.mse_loss
        with torch.no_grad():
            val_loss = 0
            for x_val, y_val in val_loader:
                outputs = self.forward(x_val)
                outputs = outputs.squeeze(1)
                val_loss += loss_fn(outputs, y_val).item()
        return torch.tensor(val_loss / len(val_loader))

    
    def save_model(self, path):
        '''
        Save the model

        Parameters:
            path: (str) the path to save the model
        '''
        torch.save(self.state_dict(), path)
        print("Model saved successfully at ", path)
