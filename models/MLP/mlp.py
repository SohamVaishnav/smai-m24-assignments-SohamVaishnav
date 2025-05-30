import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb

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
            weights[i] -= self._lr * gradients[0][len(weights)-1-i]
            biases[i] -= self._lr * gradients[1][len(weights)-1-i]
        return weights, biases

    def bgd(self, gradients_w, gradients_b, weights, biases):
        ''' 
        Batch Gradient Descent optimizer.

        Parameters:
            gradients = list of numpy arrays containing the gradients.
            weights = list of numpy arrays containing the weights.
            biases = list of numpy arrays containing the biases.
        '''
        for i in range(len(weights)):
            weights[i] -= self._lr * gradients_w[len(weights)-1-i]
            biases[i] -= self._lr * gradients_b[len(weights)-1-i]
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
        if (not verification):
            self._z = x
        if derivative:
            return np.ones_like(x)
        return x

    def sigmoid(self, x, derivative=False, verification=False):
        ''' 
        Sigmoid activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        '''
        if (not verification):
            self._z = x
        h = 1 / (1 + np.exp(-x))
        if derivative:
            return h * (1 - h)
        # self._activation = 'sigmoid'
        return h
    
    def relu(self, x, derivative=False, verification=False):
        ''' 
        ReLU activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        '''
        if (not verification):
            self._z = x
        if derivative:
            return np.where(x > 0, 1, 0)
        # self._activation = 'relu'
        return np.maximum(0, x)
    
    def tanh(self, x, derivative=False, verification=False):
        ''' 
        Tanh activation function.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        '''
        if (not verification):
            self._z = x
        if derivative:
            return 1 - np.tanh(x) ** 2
        # self._activation = 'tanh'
        return np.tanh(x)
    
    def softmax(self, x, derivative=False, verification=False):
        """
        Compute the softmax activation or its derivative.

        Parameters:
            x = numpy array containing the input data.
            derivative = boolean denoting whether to return the derivative of the function.
            verification = boolean denoting whether to test the gradient process or not
        """
        if (not verification):
            self._z = x
        exp_Z = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax_output = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        if not derivative:
            return softmax_output
        else:
            s = softmax_output
            # print((np.diag(s[0]) - np.outer(s[0], s[0])).shape)
            return np.diagflat(s) - np.dot(s, s.T)

class MultiLayerPerceptron_SingleClass(object):
    ''' 
    MultiLayerPerceptron_SingleClass class for creating a neural network model for single class 
    classification.
    '''
    def __init__(self, config) -> None:
        ''' 
        Initializes the MultiLayerPerceptron_SingleClass class.

        Parameters:
            config = dictionary containing the configuration for the model.
        '''
        self._config = config
        self._layers = []
        self._weights = []
        self._biases = []
        self._activations = self._config['activations']
        self._loss = []
        self._optimizer = self._config['optimizer']
        self._metrics = []
        self._history = None
        self._input_shape = None
        self._output_shape = None
        self._batch_size = self._config['batch_size']
        self._epochs = self._config['epochs']
        self._learning_rate = self._config['learning_rate']
        self._labels = self._config['labels']
        self._a = []
        self._y_pred = None
        self._grad_verify = self._config['grad_verify']
        self._wb = self._config['wb']

    def add(self):
        '''
        Adds a layer to the model.

        Parameters:
            layer = Layer object containing the layer details.
        '''
        layers = self._config['layers']
        activations = []
        activations.append('softmax')
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

    def fit(self, X, y, X_valid, y_valid, grad_verify=False):
        '''
        Fits the model to the data.

        Parameters:
            X = numpy array containing the input data.
            y = numpy array containing the output data.
            grad_verify = boolean denoting whether to verify the gradients or not.
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
                self._weights.append(np.random.randn(self._input_shape, self._layers[i]._units)/np.sqrt((self._input_shape+self._layers[i]._units)))
                self._biases.append(np.zeros((1, self._layers[i]._units)))
            else:
                self._weights.append(np.random.randn(self._layers[i-1]._units, self._layers[i]._units)/np.sqrt((self._layers[i-1]._units+self._layers[i]._units)))
                self._biases.append(np.zeros((1, self._layers[i]._units)))

            self._activations.append(self._layers[i]._activation)

        self._y_pred = self.train(X, y, self._labels)

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
            labels = list containing the unique labels.
        '''
        y_pred = self.predict(X)
        results = {'loss':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1_score':[]}
        metrics = Measures(y_pred, y, self._labels, True)
        results['loss'].append(self.loss(y, y_pred))
        results['accuracy'].append(metrics.accuracy())
        results['precision'].append(metrics.precision()[0])
        results['recall'].append(metrics.recall()[0])
        results['f1_score'].append(metrics.f1_score()[0])
        return results
    
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
                metrics = Measures(y_pred_train, y_shuffled, labels, True)
                history['accuracy'].append(metrics.accuracy())
                history['precision'].append(metrics.precision()[0])
                history['recall'].append(metrics.recall()[0])
                history['f1_score'].append(metrics.f1_score()[0])
                history['loss'].append(self.loss(y_shuffled, y_pred_train))
                self._metrics.append(metrics)
                
                if (self._wb):      
                    wandb.log({'train loss': self.loss(y_shuffled, y_pred_train)})
                    wandb.log({'train accuracy': metrics.accuracy()})
                    wandb.log({'train precision': metrics.precision()[0]})
                    wandb.log({'train recall': metrics.recall()[0]})
                    wandb.log({'train f1_score': metrics.f1_score()[0]})

                    y_pred = self.predict(self._X_valid)
                    metrics = Measures(y_pred, self._y_valid, labels, True)
                    wandb.log({'val loss': self.loss(self._y_valid, y_pred)})
                    wandb.log({'val accuracy': metrics.accuracy()})
                    wandb.log({'val precision': metrics.precision()[0]})
                    wandb.log({'val recall': metrics.recall()[0]})
                    wandb.log({'val f1_score': metrics.f1_score()[0]})
                    wandb.log({'epoch': epoch})

        elif (self._optimizer == 'bgd'):
            optimizer = Optimizer(self._learning_rate)
            for epoch in range(self._epochs):
                self._istraining = True
                print("Epoch: ", epoch)
                history['epoch'].append(epoch)
                y_pred = np.zeros((X.shape[0], self._output_shape))
                loss = 0
                grad_w = [np.zeros((self._weights[i].shape)) for i in range(len(self._layers))][::-1]
                grad_b = [np.zeros((self._biases[i].shape)) for i in range(len(self._layers))][::-1]
                for i in range(0, X.shape[0]):
                    y_pred[i] = self.predict(X[i:i+1])
                    loss = self.loss(y[i], y_pred[i])

                    grads = self.backprop(y[i], y_pred[i])
                    grad_w = [x+y for x, y in zip(grad_w, grads[0])]
                    grad_b = [x+y for x, y in zip(grad_b, grads[1])]
                grad_w = [grad_w[i]/X.shape[0] for i in range(len(self._layers))]
                self._weights, self._biases = optimizer.bgd(grad_w, grad_b, self._weights, self._biases)

                self._istraining = False
                y_pred_train = self.predict(X)
                metrics = Measures(y_pred_train, y, labels, True)
                history['accuracy'].append(metrics.accuracy())
                history['precision'].append(metrics.precision()[0])
                history['recall'].append(metrics.recall()[0])
                history['f1_score'].append(metrics.f1_score()[0])
                history['loss'].append(self.loss(y, y_pred_train))
                self._metrics.append(metrics)

                if (self._wb):      
                    wandb.log({'train loss': self.loss(y, y_pred_train)})
                    wandb.log({'train accuracy': metrics.accuracy()})
                    wandb.log({'train precision': metrics.precision()[0]})
                    wandb.log({'train recall': metrics.recall()[0]})
                    wandb.log({'train f1_score': metrics.f1_score()[0]})

                    y_pred = self.predict(self._X_valid)
                    metrics = Measures(y_pred, self._y_valid, labels, True)
                    wandb.log({'val loss': self.loss(self._y_valid, y_pred)})
                    wandb.log({'val accuracy': metrics.accuracy()})
                    wandb.log({'val precision': metrics.precision()[0]})
                    wandb.log({'val recall': metrics.recall()[0]})
                    wandb.log({'val f1_score': metrics.f1_score()[0]})
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
                loss = 0
                for i in range(0, X.shape[0], self._batch_size):
                    X_batch = X[i:i+self._batch_size]
                    y_batch = y[i:i+self._batch_size]

                    y_pred = self.predict(X_batch)
                    loss += self.loss(y_batch, y_pred)

                    gradients = self.backprop(y_batch, y_pred)
                    self._weights, self._biases = optimizer.mini_bgd(gradients, self._weights, self._biases, self._batch_size)
                    j += 1
                    history['batch'].append(j)

                self._istraining = False
                y_pred_train = self.predict(X)
                metrics = Measures(y_pred_train, y, labels, True)
                history['accuracy'].append(metrics.accuracy())
                history['precision'].append(metrics.precision()[0])
                history['recall'].append(metrics.recall()[0])
                history['f1_score'].append(metrics.f1_score()[0])
                history['loss'].append(self.loss(y, y_pred_train))
                self._metrics.append(metrics)
                
                if (self._wb):      
                    wandb.log({'train loss': self.loss(y, y_pred_train)})
                    wandb.log({'train accuracy': metrics.accuracy()})
                    wandb.log({'train precision': metrics.precision()[0]})
                    wandb.log({'train recall': metrics.recall()[0]})
                    wandb.log({'train f1_score': metrics.f1_score()[0]})

                    y_pred = self.predict(self._X_valid)
                    metrics = Measures(y_pred, self._y_valid, labels, True)
                    wandb.log({'val loss': self.loss(self._y_valid, y_pred)})
                    wandb.log({'val accuracy': metrics.accuracy()})
                    wandb.log({'val precision': metrics.precision()[0]})
                    wandb.log({'val recall': metrics.recall()[0]})
                    wandb.log({'val f1_score': metrics.f1_score()[0]})
                    wandb.log({'epoch': epoch})
        self._history = history
        return y_pred
    
    def forward(self, X, ver_index = 0, weights_if_verify=None):
        ''' 
        Forward pass of the model.

        Parameters:
            X = numpy array containing the input data.
            ver_index = integer denoting the index of the layer for verification.
            weights_if_verify = numpy array containing the weights for gradient verification.
        '''
        if (weights_if_verify is None):
            self._a = []
            self._a.append(X)
            for i in range(len(self._layers)):
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

        elif (weights_if_verify is not None and self._grad_verify):
            a = X
            for i in range(ver_index, len(self._layers)):
                z = np.dot(a.T, weights_if_verify[i]) + self._biases[i]
                if (self._activations[i] == 'sigmoid'):
                    a = self._layers[i].sigmoid(x = z, verification=self._grad_verify).T
                elif (self._activations[i] == 'relu'):
                    a = self._layers[i].relu(x = z, verification=self._grad_verify).T
                elif (self._activations[i] == 'tanh'):
                    a = self._layers[i].tanh(x = z, verification=self._grad_verify).T
                elif (self._activations[i] == 'softmax'):
                    a = self._layers[i].softmax(x = z, verification=self._grad_verify).T
                elif (self._activations[i] == 'linear'):
                    a = self._layers[i].linear(x = z, verification=self._grad_verify).T
            return a

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
            else:
                z = self._layers[i]._z
                if (self._activations[i] == 'sigmoid'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].sigmoid(x = z, derivative=True)
                elif (self._activations[i] == 'relu'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].relu(x = z, derivative=True)
                elif (self._activations[i] == 'tanh'):
                    error = np.dot(error, self._weights[i+1].T) * self._layers[i].tanh(x = z, derivative=True)
                elif (self._activations[i] == 'linear'):
                    error = np.dot(error, self._weights[i+1].T)

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
                        weights[j, k] -= epsilon
                        self._weights[i] = weights
                        y_pred = self.forward(a, i, self._weights)
                        loss2 = self.loss(y, y_pred)
                        gradients_verify.append((loss1 - loss2) / (2 * epsilon))
                # print(gradients_verify, "\n\n")
                # print(np.linalg.norm(gradients_verify - gradients[len(self._layers)-1-i].flatten())/np.linalg.norm(gradients_verify+gradients[len(self._layers)-1-i].flatten()))
                self._weights[i] = temp

        return gradients_w, gradients_b

    def loss(self, y_true, y_pred, derivative=False):
        ''' 
        Loss function.

        Parameters:
            y_true = numpy array containing the true output data.
            y_pred = numpy array containing the predicted output data.
            derivative = boolean denoting whether to return the derivative of the function.
        '''
        y_pred = y_pred.clip(1e-10, 1-1e-10)
        if (self._optimizer == 'mini_bgd'):
            y_true = y_true
        elif (self._istraining):
            y_true = y_true.reshape(1, -1)
        if (derivative):
            return y_pred - y_true
        return np.sum(y_true * np.log(y_pred + 1e-10))/y_pred.shape[0]
    
    def summary(self):
        ''' 
        Prints the summary of the model.
        '''
        print("Model: MultiLayerPerceptron_SingleClass")
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
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['accuracy'], mode='lines', name='accuracy'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['f1_score'], mode='lines', name='f1_score'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['precision'], mode='lines', name='precision'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self._history['epoch'], y=self._history['recall'], mode='lines', name='recall'), row=2, col=1)
        fig.update_layout(title='Model History')
        fig.show()
        return None

    