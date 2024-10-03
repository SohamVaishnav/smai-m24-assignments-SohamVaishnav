import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('mlp.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.MLP.mlp import *
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
            weights[i] -= self.lr * gradients[i]
            biases[i] -= self.lr * gradients[i].mean()
        return weights, biases

    def bgd(self, gradients, weights, bias):
        ''' 
        Batch Gradient Descent optimizer.

        Parameters:
            gradients = list of numpy arrays containing the gradients.
            weights = list of numpy arrays containing the weights.
            biases = list of numpy arrays containing the biases.
        '''
        grad = np.mean(gradients, axis=0)
        for i in range(len(weights)):
            weights[i] -= self.lr * grad
            bias[i] -= self.lr * grad.mean()
        return weights, bias
        
    def mini_bgd(self, gradients, weights, biases, batch_size):
        ''' 
        Mini Batch Gradient Descent optimizer.

        Parameters:
            gradients = list of numpy arrays containing the gradients.
            weights = list of numpy arrays containing the weights.
            biases = list of numpy arrays containing the biases.
            batch_size = integer denoting the batch size.
        '''
        grad = np.mean(gradients, axis=0)
        for i in range(len(weights)):
            weights[i] -= self.lr * grad            
            biases[i] -= self.lr * grad.mean()
        return weights, biases

class Layer(object):
    ''' 
    Layer class for creating a layer in the neural network model.
    '''
    def __init__(self, units) -> None:
        '''
        Initializes the Layer class.

        Parameters:
            units = integer denoting the number of units in the layer.
        '''
        self._units = units
        self._z = None #the inputs to the layer

    def sigmoid(self, x, derivative=False):
        ''' 
        Sigmoid activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
        '''
        self._z = x
        h = 1 / (1 + np.exp(-x))
        if derivative:
            return h * (1 - h)
        self._activation = 'sigmoid'
        return h
    
    def relu(self, x, derivative=False):
        ''' 
        ReLU activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
        '''
        self._z = x
        if derivative:
            return np.where(x > 0, 1, 0)
        self._activation = 'relu'
        return np.maximum(0, x)
    
    def tanh(self, x, derivative=False):
        ''' 
        Tanh activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
        '''
        self._z = x
        if derivative:
            return 1 - np.tanh(x) ** 2
        self._activation = 'tanh'
        return np.tanh(x)

class MultiLayerPerceptron_SingleClass(object):
    ''' 
    MultiLayerPerceptron_SingleClass class for creating a neural network model for single class 
    classification.
    '''
    def __init__(self) -> None:
        ''' 
        Initializes the MultiLayerPerceptron_SingleClass class.
        '''
        self._layers = []
        self._weights = []
        self._biases = []
        self._activations = []
        self._loss = None
        self._optimizer = None
        self._metrics = []
        self._history = None
        self._input_shape = None
        self._output_shape = None
        self._batch_size = None
        self._epochs = None
        self._learning_rate = None
        self._labels = None
        self._a = []

    def add(self, layer: Layer):
        '''
        Adds a layer to the model.

        Parameters:
            layer = Layer object containing the layer details.
        '''
        self._layers.append(layer)
        
    def setHyperParams(self, loss, optimizer, metrics, lr):
        '''
        Compiles the model with the loss, optimizer and metrics.

        Parameters:
            loss = function denoting the loss function.
            optimizer = function denoting the optimizer.
            metrics = list of functions denoting the metrics.
            lr = float denoting the learning rate.
        '''
        self._loss = loss
        self._optimizer = optimizer
        self._metrics = metrics
        self._learning_rate = lr

    def fit(self, X, y, batch_size, epochs):
        '''
        Fits the model to the data.

        Parameters:
            X = numpy array containing the input data.
            y = numpy array containing the output data.
            batch_size = integer denoting the batch size.
            epochs = integer denoting the number of epochs.
        '''

        self._input_shape = X.shape[1]
        self._output_shape = y.shape[1]
        self._batch_size = batch_size
        self._epochs = epochs

        for i in range(len(self._layers)):
            if i == 0:
                self._weights.append(np.random.randn(self._input_shape, self._layers[i].units))
                self._biases.append(np.random.randn(self._layers[i].units))
            else:
                self._weights.append(np.random.randn(self._layers[i-1].units, self._layers[i].units))
                self._biases.append(np.random.randn(self._layers[i].units))

            self._activations.append(self._layers[i]._activation)

        self._history = self._train(X, y)

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
        return self.loss(y, y_pred)
    
    def train(self, X, y, labels):
        ''' 
        Trains the model on the input data.

        Parameters:
            X = numpy array containing the input data.
            y = numpy array containing the output data.
            labels = list containing the unique labels.
        '''
        history = {'epoch': [], 'loss': [], 'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []}
        if (self._optimizer == 'sgd'):
            optimizer = Optimizer(self._learning_rate)
            for epoch in range(self._epochs):
                history['epoch'].append(epoch)
                y_pred = []
                for i in range(0, X.shape[0]):
                    y_pred.append(self.predict(X[i]))
                    loss = self.loss(y[i], y_pred[i])
                    history['loss'].append(loss)
                    
                    gradients = self.backprop(y[i], y_pred[i])
                    self._weights, self._biases = optimizer.sgd(gradients, self._weights, self._biases)
                metrics = Measures(y_pred, y, labels)
                history['accuracy'].append(metrics.accuracy())
                history['f1_score'].append(metrics.f1_score()[0])
                history['precision'].append(metrics.precision()[0])
                history['recall'].append(metrics.recall()[0])
                self._metrics.append(metrics)


        elif (self._optimizer == 'bgd'):
            optimizer = Optimizer(self._learning_rate)
            for epoch in range(self._epochs):
                history['epoch'].append(epoch)
                y_pred = []
                for i in range(0, X.shape[0]):
                    y_pred.append(self.predict(X[i]))
                    loss = self.loss(y[i], y_pred)
                    history['loss'].append(loss)

                gradients = self.backprop(y, y_pred)
                self._weights, self._biases = optimizer.bgd(gradients, self._weights, self._biases)
                metrics = Measures(y_pred, y, labels)
                history['accuracy'].append(metrics.accuracy())
                history['f1_score'].append(metrics.f1_score()[0])
                history['precision'].append(metrics.precision()[0])
                history['recall'].append(metrics.recall()[0])
                self._metrics.append(metrics)
        
        elif (self._optimizer == 'mini_bgd'):
            history.update({'batch_size': self._batch_size})
            history['batch'] = []
            optimizer = Optimizer(self._learning_rate)
            for epoch in range(self._epochs):
                history['epoch'].append(epoch)
                j = 1
                for i in range(0, X.shape[0], self._batch_size):
                    X_batch = X[i:i+self._batch_size]
                    y_batch = y[i:i+self._batch_size]

                    y_pred = self.predict(X_batch)
                    loss = self.loss(y_batch, y_pred)
                    history['loss'].append(loss)

                    gradients = self.backprop(y_batch, y_pred)
                    self._weights, self._biases = optimizer.mini_bgd(gradients, self._weights, self._biases, self._batch_size)
                    metrics = Measures(y_pred, y_batch, labels)
                    history['accuracy'].append(metrics.accuracy())
                    history['f1_score'].append(metrics.f1_score()[0])
                    history['precision'].append(metrics.precision()[0])
                    history['recall'].append(metrics.recall()[0])
                    history['batch'].append(j)
                    self._metrics.append(metrics)
                    j += 1
        self._history = history
        return y_pred
    
    def forward(self, X):
        ''' 
        Forward pass of the model.

        Parameters:
            X = numpy array containing the input data.
        '''
        self._a.append(X)
        for i in range(len(self._layers)):
            z = np.dot(self._a, self._weights[i]) + self._biases[i]
            if (self._activations[i] == 'sigmoid'):
                self._a.append(self._layers[i].sigmoid(z))
            elif (self._activations[i] == 'relu'):
                self._a.append(self._layers[i].relu(z))
            elif (self._activations[i] == 'tanh'):
                self._a.append(self._layers[i].tanh(z))
        return self._a[-1]
    
    def backprop(self, y, y_pred, verify: bool = False):
        '''
        Backward propagation of the model.

        Parameters:
            y = numpy array containing the output data.
            y_pred = numpy array containing the predicted output data.
            verify = boolean denoting whether to test the gradient process or not
        '''
        gradients = []
        for i in range(len(self._layers)-1, -1, -1):
            if i == len(self._layers)-1:
                z = self._layers[i]._z
                if (self._activations[i] == 'sigmoid'):
                    error = self.loss(y, y_pred, derivative=True) * self._layers[i].sigmoid(z, derivative=True)
                elif (self._activations[i] == 'relu'):
                    error = self.loss(y, y_pred, derivative=True) * self._layers[i].relu(z, derivative=True)
                elif (self._activations[i] == 'tanh'):
                    error = self.loss(y, y_pred, derivative=True) * self._layers[i].tanh(z, derivative=True)
            else:
                z = self._layers[i]._z
                if (self._activations[i] == 'sigmoid'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].sigmoid(z, derivative=True)
                elif (self._activations[i] == 'relu'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].relu(z, derivative=True)
                elif (self._activations[i] == 'tanh'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].tanh(z, derivative=True)

            # if (self._activations[i-1] == 'sigmoid'):
            #     gradients.append(np.dot(self._layers[i-1].sigmoid(y_pred, derivative=True).T, error))
            # elif (self._activations[i-1] == 'relu'):
            #     gradients.append(np.dot(self._layers[i-1].relu(y_pred, derivative=True).T, error))
            # elif (self._activations[i-1] == 'tanh'):
            #     gradients.append(np.dot(self._layers[i-1].tanh(y_pred, derivative=True).T, error))
            if i > 0:
                gradients.append(np.dot(self._a[i-1].T, error))
            
            if (verify):
                gradients_verify = []
                epsilon = 1e-5
                weights = self._weights[i]
                for j in range(weights.shape[0]):
                    for k in range(weights.shape[1]):
                        weights[j, k] += epsilon
                        y_pred = self.forward(self._a[i-1])
                        loss1 = self.loss(y, y_pred)
                        weights[j, k] -= 2 * epsilon
                        y_pred = self.forward(self._a[i-1])
                        loss2 = self.loss(y, y_pred)
                        gradients_verify.append((loss1 - loss2) / (2 * epsilon))
            
                print(np.allclose(gradients[i], gradients_verify))

        return gradients

    def loss(self, y_true, y_pred, derivative=False):
        ''' 
        Loss function.

        Parameters:
            y_true = numpy array containing the true output data.
            y_pred = numpy array containing the predicted output data.
            derivative = boolean denoting whether to return the derivative of the function.
        '''
        if derivative:
            return y_pred - y_true
        return np.mean((y_pred - y_true) ** 2)