import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('mlp_regression.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.MLP.mlp_regression import *
from performance_measures.metricsMLP import *

class Optimizer(object):
    def __init__(self, lr) -> None:
        self._lr = lr

    def sgd(self, gradients, weights, biases):
        ''' 
        Stochastic Gradient Descent optimizer.

        Parameters:
            gradients = list of numpy arrays containing the gradients.
            weights = list of numpy arrays containing the weights.
            biases = list of numpy arrays containing the biases.
        '''
        for i in range(len(weights)):
            weights[i] -= self._lr * gradients[0][len(weights)-1-i]
            biases[i] -= self._lr * gradients[1][len(weights)-1-i]
        return weights, biases

    def bgd(self, grad_w, grad_b, weights, biases):
        ''' 
        Batch Gradient Descent optimizer.

        Parameters:
            gradients = list of numpy arrays containing the gradients.
            weights = list of numpy arrays containing the weights.
            biases = list of numpy arrays containing the biases.
        '''
        for i in range(len(weights)):
            weights[i] -= self._lr * grad_w[len(weights)-1-i]
            biases[i] -= self._lr * grad_b[len(weights)-1-i]
        return weights, biases
        
    def mini_bgd(self, gradients, weights, biases, batch_size):
        ''' 
        Mini Batch Gradient Descent optimizer.

        Parameters:
            gradients = list of numpy arrays containing the gradients.
            weights = list of numpy arrays containing the weights.
            biases = list of numpy arrays containing the biases.
            batch_size = integer denoting the batch size.
        '''
        for i in range(len(weights)):
            weights[i] -= self._lr * gradients[0][len(weights)-1-i]/batch_size            
            biases[i] -= self._lr * gradients[1][len(weights)-1-i]/batch_size
        return weights, biases

class Layer(object):
    ''' 
    Layer class for creating a layer in the neural network model.
    '''
    def __init__(self, units, activation_func) -> None:
        '''
        Initializes the Layer class.

        Parameters:
            units = integer denoting the number of units in the layer.
            activation_func = string denoting the activation function.
        '''
        self._units = units
        self._z = None #the inputs to the layer
        self._activation = activation_func

    def linear(self, x, derivative=False, verification=False):
        ''' 
        Linear activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        '''
        if (not verification and not derivative):
            self._z = x
        if derivative:
            return 1
        return x

    def sigmoid(self, x, derivative=False, verification=False):
        ''' 
        Sigmoid activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        '''
        if (not verification and not derivative):
            self._z = x
        h = 1 / (1 + np.exp(-x))
        if derivative:
            return h * (1 - h)
        return h
    
    def relu(self, x, derivative=False, verification=False):
        ''' 
        ReLU activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        '''
        if (not verification and not derivative):
            self._z = x
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)
    
    def tanh(self, x, derivative=False, verification=False):
        ''' 
        Tanh activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        '''
        if (not verification and not derivative):
            self._z = x
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)
    
    def softmax(self, x, derivative=False, verification=False):
        """
        Compute the softmax activation or its derivative.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        """
        if (not verification and not derivative):
            self._z = x
        exp_Z = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax_output = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        if not derivative:
            return softmax_output
        else:
            s = softmax_output
            return np.diagflat(s) - np.dot(s, s.T)

class MutliLayerPerceptron_Regression(object):
    ''' 
    MutliLayerPerceptron_Regression is a class that implements a multi layer perceptron for regression tasks.
    '''
    def __init__(self, config) -> None:
        ''' 
        Initializes the MutliLayerPerceptron_Regression class.

        Parameters:
            config = dictionary containing the configuration for the model.
        '''
        self._config = config
        self._layers = []
        self._weights = []
        self._biases = []
        self._activations = config['activations']
        self._loss = config['loss']
        self._optimizer = config['optimizer']
        self._metrics = []
        self._history = None
        self._input_shape = None
        self._output_shape = None
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']
        self._learning_rate = config['learning_rate']
        self._a = []
        self._y_pred = None
        self._grad_verify = config['grad_verify']
        self._wb = config['wb']
        
    def add(self):
        '''
        Adds a layer to the model.

        Parameters:
            layer = Layer object containing the layer details.
        '''
        layers = self._config['layers']
        activations = []
        if (self._loss == 'cross_entropy'):
            activations.append('sigmoid')
        else:
            activations.append('linear')
        for i in range(len(layers)-1):
            activations.insert(0, self._activations)
        self._activations = activations
        for i in range(len(layers)):
            self._layers.append(Layer(layers[i], activations[i]))
        
    def setHyperParams(self, hyperparams: dict):
        '''
        Compiles the model with the loss, optimizer and metrics.

        Parameters:
            hyperparams = dictionary containing the hyperparameters for the model.
        '''
        self._optimizer = hyperparams['optimizer']
        self._batch_size = hyperparams['batch_size']
        self._learning_rate = hyperparams['learning_rate']
        self._epochs = hyperparams['epochs']
        self._loss = hyperparams['loss']

    def fit(self, X, y, X_valid, y_valid, grad_verify=False):
        '''
        Fits the model to the data.

        Parameters:
            X = numpy array containing the input data.
            y = numpy array containing the output data.
            batch_size = integer denoting the batch size.
            epochs = integer denoting the number of epochs.
        '''
        self._activations = []
        self._input_shape = X.shape[1]
        self._output_shape = y.shape[1]
        self._data_points = X.shape[0]

        self._X_valid = X_valid
        self._y_valid = y_valid

        if (grad_verify):
            self._grad_verify = True

        for i in range(len(self._layers)):
            if i == 0:
                self._weights.append(np.random.randn(self._input_shape, self._layers[i]._units)*np.sqrt(2/(self._input_shape+self._layers[i]._units)))
                self._biases.append(np.zeros((1, self._layers[i]._units)))
            else:
                self._weights.append(np.random.randn(self._layers[i-1]._units, self._layers[i]._units)*np.sqrt(2/(self._layers[i-1]._units+self._layers[i]._units)))
                self._biases.append(np.zeros((1, self._layers[i]._units)))

            self._activations.append(self._layers[i]._activation)

        self._y_pred = self.train(X, y)

    def predict(self, X):
        ''' 
        Predicts the output for the input data.

        Parameters:
            X = numpy array containing the input data.
        '''
        return self.forward(X)

    def evaluate(self, X, y):
        ''' 
        Evaluates the model on the input data.

        Parameters:
            X = numpy array containing the input data.
            y = numpy array containing the output data.
        '''
        y_pred = self.predict(X)
        results = {'loss':[], 'mse':[], 'rmse':[], 'r2':[]}
        metrics = metrics_regression(y_pred, y)
        results['loss'] = self.loss(y, y_pred)
        results['mse'] = metrics.mse()
        results['rmse'] = metrics.rmse()
        results['r2'] = metrics.r2()
        return results
    
    def train(self, X, y):
        ''' 
        Trains the model on the input data.

        Parameters:
            X = numpy array containing the input data.
            y = numpy array containing the output data.
            labels = list containing the unique labels.
        '''
        history = {'epoch': [], 'loss': [], 'mse': [], 'rmse': [], 'r2': []}
        if (self._optimizer == 'sgd'):
            optimizer = Optimizer(self._learning_rate)
            for epoch in range(self._epochs):
                self._istraining = True
                print("Epoch: ", epoch)
                history['epoch'].append(epoch)
                y_pred = np.zeros((X.shape[0], self._output_shape))
                indices = np.arange(0, X.shape[0])
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                loss = 0
                for i in range(0, X.shape[0]):
                    y_pred[i] = self.predict(X_shuffled[i:i+1])
                    loss += self.loss(y_shuffled[i], y_pred[i])

                    gradients = self.backprop(y_shuffled[i], y_pred[i])
                    self._weights, self._biases = optimizer.sgd(gradients, self._weights, self._biases)

                self._istraining = False
                y_pred_train = self.predict(X_shuffled)
                metrics = metrics_regression(y_pred_train, y_shuffled)
                history['mse'].append(metrics.mse())
                history['rmse'].append(metrics.rmse())
                history['r2'].append(metrics.r2())
                history['loss'].append(self.loss(y_shuffled, y_pred_train))
                self._metrics.append(metrics)
                
                if (self._wb):
                    wandb.log({'train loss': self.loss(y_shuffled, y_pred_train)})
                    wandb.log({'train mse': metrics.mse()})
                    wandb.log({'train rmse': metrics.rmse()})
                    wandb.log({'train r2': metrics.r2()})

                    y_pred = self.predict(self._X_valid)
                    metrics = metrics_regression(y_pred, self._y_valid)
                    wandb.log({'val loss': self.loss(self._y_valid, y_pred)})
                    wandb.log({'val mse': metrics.mse()})
                    wandb.log({'val rmse': metrics.rmse()})
                    wandb.log({'val r2': metrics.r2()})
                    wandb.log({'epoch': epoch})

        elif (self._optimizer == 'bgd'):
            optimizer = Optimizer(self._learning_rate)
            for epoch in range(self._epochs):
                self._istraining = True
                print("Epoch: ", epoch)
                history['epoch'].append(epoch)
                y_pred = np.zeros((X.shape[0], self._output_shape))
                grad_w = [np.zeros((self._weights[i].shape)) for i in range(len(self._layers))][::-1]
                grad_b = [np.zeros((self._biases[i].shape)) for i in range(len(self._layers))][::-1]
                loss = 0
                gradients = [np.zeros((self._weights[i].shape)) for i in range(len(self._layers))][::-1]
                for i in range(0, X.shape[0]):
                    y_pred[i] = self.predict(X[i:i+1])
                    loss += self.loss(y[i], y_pred[i])
                    grads = self.backprop(y[i], y_pred[i])
                    grad_w = [x+y for x, y in zip(grad_w, grads[0])]
                    grad_b = [x+y for x, y in zip(grad_b, grads[1])]
                grad_w = [grad_w[i]/X.shape[0] for i in range(len(self._layers))]
                self._weights, self._biases = optimizer.bgd(grad_w, grad_b, self._weights, self._biases)

                self._istraining = False
                y_pred_train = self.predict(X)
                metrics = metrics_regression(y_pred_train, y)
                history['mse'].append(metrics.mse())
                history['rmse'].append(metrics.rmse())
                history['r2'].append(metrics.r2())
                history['loss'].append(self.loss(y, y_pred_train))
                self._metrics.append(metrics)

                if (self._wb):
                    wandb.log({'train loss': self.loss(y, y_pred_train)})
                    wandb.log({'train mse': metrics.mse()})
                    wandb.log({'train rmse': metrics.rmse()})
                    wandb.log({'train r2': metrics.r2()})

                    y_pred = self.predict(self._X_valid)
                    metrics = metrics_regression(y_pred, self._y_valid)
                    wandb.log({'val loss': self.loss(self._y_valid, y_pred)})
                    wandb.log({'val mse': metrics.mse()})
                    wandb.log({'val rmse': metrics.rmse()})
                    wandb.log({'val r2': metrics.r2()})
                    wandb.log({'epoch': epoch})
        
        elif (self._optimizer == 'mini_bgd'):
            history.update({'batch_size': self._batch_size})
            history['batch'] = []
            optimizer = Optimizer(self._learning_rate)
            for epoch in range(self._epochs):
                self._istraining = True
                print("Epoch: ", epoch)
                history['epoch'].append(epoch)
                j = 0
                for i in range(0, X.shape[0], self._batch_size):
                    X_batch = X[i:i+self._batch_size]
                    y_batch = y[i:i+self._batch_size]

                    y_pred = self.predict(X_batch)
                    loss = self.loss(y_batch, y_pred)

                    gradients = self.backprop(y_batch, y_pred)
                    self._weights, self._biases = optimizer.mini_bgd(gradients, self._weights, self._biases, self._batch_size)
                    j += 1
                self._istraining = False
                y_pred_train = self.predict(X)
                metrics = metrics_regression(y_pred_train, y)
                history['mse'].append(metrics.mse())
                history['rmse'].append(metrics.rmse())
                history['r2'].append(metrics.r2())
                history['loss'].append(self.loss(y, y_pred_train))
                self._metrics.append(metrics)

                if (self._wb):
                    wandb.log({'train loss': self.loss(y, y_pred_train)})
                    wandb.log({'train mse': metrics.mse()})
                    wandb.log({'train rmse': metrics.rmse()})
                    wandb.log({'train r2': metrics.r2()})

                    y_pred = self.predict(self._X_valid)
                    metrics = metrics_regression(y_pred, self._y_valid)
                    wandb.log({'val loss': self.loss(self._y_valid, y_pred)})
                    wandb.log({'val mse': metrics.mse()})
                    wandb.log({'val rmse': metrics.rmse()})
                    wandb.log({'val r2': metrics.r2()})
                    wandb.log({'epoch': epoch})
        self._history = history
        return y_pred

    def forward(self, X, encode=False, decode=False):
        ''' 
        Forward pass of the model.

        Parameters:
            X = numpy array containing the input data.
            encode = boolean denoting whether to encode or decode.
        '''
        if (encode and not decode):
            self._a = []
            self._a.append(X)
            start = 0
            if (len(self._layers)%2 != 0):
                end = int((len(self._layers)-1)/2)
            else:
                end = int(len(self._layers)/2)
        elif (decode and not encode):
            self._a = []
            self._a.append(X)
            start = (len(self._layers)-1)/2 if len(self._layers)%2 != 0 else (len(self._layers))/2
            end = len(self._layers)
        elif (not encode and not decode):
            self._a = []
            self._a.append(X)
            start = 0
            end = len(self._layers)
        for i in range(start, end):
            z = np.dot(self._a[i], self._weights[i]) + self._biases[i]
            if (self._activations[i] == 'sigmoid'):
                self._a.append(self._layers[i].sigmoid(x = z))
            elif (self._activations[i] == 'relu'):
                self._a.append(self._layers[i].relu(x = z))
            elif (self._activations[i] == 'tanh'):
                self._a.append(self._layers[i].tanh(x = z))
            elif (self._activations[i] == 'softmax'):
                self._a.append(self._layers[i].softmax(x = z))
            elif (self._activations[i] == 'linear'):
                self._a.append(self._layers[i].linear(x = z))

        return self._a[-1]
    
    def backprop(self, y, y_pred):
        '''
        Backward propagation of the model.

        Parameters:
            y = numpy array containing the output data.
            y_pred = numpy array containing the predicted output data.
        '''
        gradients_w = []
        gradients_b = []
        for i in range(len(self._layers)-1, -1, -1):
            if i == len(self._layers)-1:
                z = self._layers[i]._z
                if (self._activations[i] == 'sigmoid'):
                    error = self.loss(y, y_pred, derivative=True) * self._layers[i].sigmoid(x = z, derivative=True)
                elif (self._activations[i] == 'relu'):
                    error = self.loss(y, y_pred, derivative=True) * self._layers[i].relu(x = z, derivative=True)
                elif (self._activations[i] == 'tanh'):
                    error = self.loss(y, y_pred, derivative=True) * self._layers[i].tanh(x = z, derivative=True)
                elif (self._activations[i] == 'softmax'):
                    error = self.loss(y, y_pred, derivative=True)
                elif (self._activations[i] == 'linear'):
                    error = self.loss(y, y_pred, derivative=True)
                # if (self._optimizer == 'mini_bgd'):
                #     error = np.mean(error, axis=0).reshape(1, -1)
            else:
                z = self._layers[i]._z
                if (self._activations[i] == 'sigmoid'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].sigmoid(x = z, derivative=True)
                elif (self._activations[i] == 'relu'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].relu(x = z, derivative=True)
                elif (self._activations[i] == 'tanh'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].tanh(x = z, derivative=True)

            grad_b = np.sum(error, axis=0, keepdims=True)/y.shape[0]
            gradients_b.append(grad_b)

            if i >= 0:
                if (self._optimizer == 'mini_bgd' and self._a[i].shape[0]%self._batch_size == 0):
                    a = self._a[i].reshape(-1, self._batch_size)
                else:
                    a = self._a[i].T
                gradients_w.append(a.dot(error)/y.shape[0])

            if (self._grad_verify and i >= 0):
                gradients_verify = []
                epsilon = 1e-10
                weights = self._weights[i]
                temp = self._weights[i]
                for j in range(weights.shape[0]):
                    for k in range(weights.shape[1]):
                        weights[j, k] += epsilon
                        self._weights[i] = weights
                        y_pred = self.forward(a, i, self._weights)
                        loss1 = self.loss(y, y_pred)
                        weights[j, k] -= 2*epsilon
                        self._weights[i] = weights
                        y_pred = self.forward(a, i, self._weights)
                        loss2 = self.loss(y, y_pred)
                        gradients_verify.append((loss1 - loss2) / (2 * epsilon))
                        weights[j, k] += epsilon
                # print(gradients_verify, "\n\n")
                # print(np.linalg.norm(gradients_verify - gradients[len(self._layers)-1-i].flatten())/np.linalg.norm(gradients_verify+gradients[len(self._layers)-1-i].flatten()))
                self._weights[i] = temp
        return gradients_w, gradients_b

    def encode(self, X):
        ''' 
        Encodes the input data.

        Parameters:
            X = numpy array containing the input data.
        '''
        return self.forward(X, encode=True)

    def decode(self, X):
        ''' 
        Decodes the input data.

        Parameters:
            X = numpy array containing the input data.
        '''
        return self.forward(X, decode=True)

    def loss(self, y_true, y_pred, derivative=False):
        ''' 
        Loss function.

        Parameters:
            y_true = numpy array containing the true output data.
            y_pred = numpy array containing the predicted output data.
            derivative = boolean denoting whether to return the derivative of the function.
        '''
        if ((self._optimizer == 'sgd' or self._optimizer == 'bgd') and self._istraining):
            y_pred = y_pred.reshape(1, -1)
            y_true = y_true.reshape(1, -1)
        if (self._loss == 'cross_entropy'):
            y_pred = y_pred.clip(1e-10, 1-1e-10)
            if (derivative):
                return (y_true - y_pred) / (y_pred * (1 - y_pred) + 1e-10)
            return -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))
        if (self._loss == 'mae'):
            if (derivative):
                return np.sign(y_pred - y_true)
            return np.mean(np.abs(y_true - y_pred))
        if (self._loss == 'mse'):
            if (derivative):
                return 2 * (y_pred - y_true)
            return np.mean((y_true - y_pred) ** 2)
        elif (self._loss == 'rmse'):
            if (derivative):
                return 2 * (y_pred - y_true) / y_true.size
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif (self._loss == 'R2'):
            if (derivative):
                return 2 * (y_pred - y_true) / y_true.size
            return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    def summary(self):
        ''' 
        Prints the summary of the model.
        '''
        print("Model: MultiLayerPerceptron_Regression")
        print("_________________________________________________________________")
        print("Layer (type)                 Output Shape              Param #")
        print("=================================================================")
        for i in range(len(self._layers)):
            if i == 0:
                print("Input Layer                   "+str(self._input_shape)+"                       0")
            else:
                print("Hidden Layer                  "+str(self._layers[i]._units)+"                       "+str(self._weights[i-1].size + self._biases[i-1].size))
        print("Output Layer                  "+str(self._output_shape)+"                       "+str(self._weights[-1].size + self._biases[-1].size))
        print("=================================================================")
        print("Total params: "+str(sum([self._weights[i].size + self._biases[i].size for i in range(len(self._weights))])))
        print("Trainable params: "+str(sum([self._weights[i].size + self._biases[i].size for i in range(len(self._weights))])))
        print("Non-trainable params: 0")
        print("_________________________________________________________________")
        return None
    
    def plot(self):
        ''' 
        Plots the history of the model.
        '''
        fig = sp.make_subplots(rows=2, cols=1, subplot_titles=('Loss', 'Metrics'))
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['loss'], mode='lines', name='loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['mse'], mode='lines', name='mse'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['rmse'], mode='lines', name='rmse'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['r2'], mode='lines', name='r2'), row=2, col=1)
        fig.update_layout(title='Model History')
        fig.show()
        return None
    
class metrics_regression(object):
    ''' 
    Metrics for regression tasks.
    ''' 
    def __init__(self, y_pred, y_true) -> None:
        ''' 
        Initializes the metrics_regression class.
        '''
        self._y_pred = y_pred
        self._y_true = y_true
        pass

    def mse(self):
        ''' 
        Mean Squared Error.
        '''
        return np.mean((self._y_true - self._y_pred) ** 2)
    
    def rmse(self):
        ''' 
        Root Mean Squared Error.
        '''
        return np.sqrt(np.mean((self._y_true - self._y_pred) ** 2))
    
    def r2(self):
        ''' 
        R2 Score.
        '''
        return 1 - np.sum((self._y_true - self._y_pred) ** 2) / np.sum((self._y_true - np.mean(self._y_true)) ** 2)