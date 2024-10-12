import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

import os
import sys 

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff

import time

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a3.py')))
CurrDIR = os.path.dirname(os.path.abspath('a3.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.MLP.mlp import *
from models.MLP.mlp_multi import *
from models.MLP.mlp_regression import *
from models.MLP.auto_encoder import *
from performance_measures.metricsMLP import *

from models.MLP.utils import *

from sklearn.preprocessing import StandardScaler

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/3/")

def DataLoader(DataDIR: str, datafilename: str):
    ''' 
    Loads the data from the directory. \n

    Parameters:
        DataDIR = string denoting the path to the directory containing data.
        datafilename = string denoting the name of the data file that needs to be read from DataDIR
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)

    data = pd.read_csv(data_path)
    print("Data has been loaded into a dataframe successfully!")
    return data

def Word2Num(dataset: pd.DataFrame, string_features) -> pd.DataFrame:
    ''' 
    Converts the string entries into some number which can then be used for classification. 
    
    Arguments:
    dataset = the original data that the model is dealing with
    string_features = the columns from the dataset that contain string entries
    '''
    for i in string_features:
        words, count = np.unique(dataset[i], return_counts = True)
        nums = count/np.max(count)
        encodings = {value: nums[i] for i, value in enumerate(words)}
        dataset[i] = dataset[i].replace(encodings)
    return dataset

def DataWriter(Data: pd.DataFrame, DataDIR: str, datafilename: str):
    ''' 
    Writes the data to the directory. \n
    
    Parameters:
        Data = pandas dataframe containing the data.
        DataDIR = string denoting the path to the directory containing data.
        datafilename = string denoting the name of the data file that needs to be written to DataDIR
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)

    Data.to_csv(data_path, index = False)
    print("Data has been written to "+data_path+" successfully!")
    return data_path

def DataPreprocess(data: pd.DataFrame, isMulti: bool = False, isReg: bool = False) -> pd.DataFrame:
    ''' 
    Preprocesses the data.

    Parameters:
        data = pandas dataframe containing the data.
        isMulti = boolean denoting whether the data is for multi class classification or not.
        isReg = boolean denoting whether the data is for regression or not.
    '''
    data = data.dropna()
    temp = None
    if (not isMulti and not isReg):
        data = data.drop_duplicates(subset = ['Id'], keep = 'first')
        data = data.drop(columns = ['Id'])

        if ('quality' in data.columns):
            temp = data['quality']
            data = data.drop(columns = ['quality'])
    
    elif (isMulti and not isReg):
        if ('label' in data.columns):
            temp = data['label']
            data = data.drop(columns = ['label'])
    
    elif (isReg):
        if ('MEDV' in data.columns):
            temp = data['MEDV']
            data = data.drop(columns = ['MEDV'])

    for i in data.columns:
        if (type(data[i].iloc[0]) == str or data[i].iloc[0].dtype == bool):
            continue
        means = np.mean(data[i], axis = 0)
        vars = np.var(data[i], axis = 0)
        data[i] = (data[i] - means)/np.sqrt(vars)
    
    print("Data has been preprocessed successfully!")
    if (temp is not None):
        data = pd.concat([data, temp], axis = 1)
    return data

################################### MLP ###################################
# def createMLP(layers: list, activations: list, hyperparams: dict, isSingleClass: bool = True):
#     ''' 
#         Creates a MLP model.

#         Parameters:
#             layers = list containing the number of neurons (as integers) in each layer.
#             activations = list containing the activation functions (as strings) for each layer.
#             hyperparams = dictionary containing the hyperparameters for the model. The keys are:
#                 learning_rate = float denoting the learning rate for the model.
#                 epochs = integer denoting the number of epochs for the model.
#                 batch_size = integer denoting the batch size for the model.
#                 optimizer = string denoting the optimizer for the model ('sgd', 'bgd', mini_bgd').
#             isSingleClass = boolean denoting whether the model is for single class classification or not.
#     '''
#     if (isSingleClass):
#         model = MultiLayerPerceptron_SingleClass()
#         for i in range(len(layers)):
#             layer = Layer(layers[i], activations[i])
#             model.add(layer)
#         model.setHyperParams(hyperparams)
#         # wandb.init(project = "wine_quality_MLP", config = {'layers': layers, 'activations': activations, 
#         # 'learning_rate': hyperparams['learning_rate'], 'epochs': hyperparams['epochs'], 
#         # 'batch_size': hyperparams['batch_size'], 'optimizer': hyperparams['optimizer']})

#     else:
#         model = MultiLayerPerceptron_MultiClass()
#         for i in range(len(layers)):
#             layer = Layer(layers[i], activations[i])
#             model.add(layer)
#         model.setHyperParams(hyperparams)
#         # wandb.init(project = "wine_quality_MLP", config = {'layers': layers, 'activations': activations, 
#         # 'learning_rate': hyperparams['learning_rate'], 'epochs': hyperparams['epochs'], 
#         # 'batch_size': hyperparams['batch_size'], 'optimizer': hyperparams['optimizer']})

#     return model

# def runMLP(model, data: pd.DataFrame, grad_verify: bool = False, isMulti: bool = False):
#     ''' 
#         Runs the MLP model.

#         Parameters:
#             model = MultiLayerPerceptron class object.
#             data = pandas dataframe containing the data.
#             grad_verify = boolean denoting whether to verify the gradients or not
#             isMulti = boolean denoting whether the model is for multi class classification or not.
#     '''
#     if (not isMulti):
#         X = data.drop(columns = ['quality'])
#         y = data['quality']
#         num_classes = len(np.unique(y))
#         labels = np.unique(y)-3
#         Y = np.zeros((y.shape[0], num_classes))
#         for i in range(y.shape[0]):
#             Y[i, y.iloc[i]-3] = 1
#         model.fit(X.to_numpy(), Y, labels, grad_verify)
#     else:
#         X = data.drop(columns = ['labels'])
#         y = data['labels']
#         labels = []
#         for i in y:
#             temp = i.split(' ')
#             for j in temp:
#                 labels.append(j)
#         labels = np.unique(labels)
#         num_classes = len(labels)
#         Y = np.zeros((y.shape[0], num_classes))
#         for i in range(y.shape[0]):
#             temp = y.iloc[i].split(' ')
#             for j in temp:
#                 Y[i, np.where(labels == j)] = 1
#         model.fit(X.to_numpy(), Y, labels)
#         model.plot()

#     return None

################################### Single label classification ###################################
# wandb.login()

# config_sweep = {
# 'method': 'bayes',
# 'name': 'Hyperparameter tuning: Single label classification', 
# 'metric': {
#     'goal': 'maximize',
#     'name': 'accuracy'
# }, 
# 'parameters': {
#     'epochs': {'values': [50, 100, 500, 1000]},
#     'layers': {'values': [[6], [15, 6], [64, 32, 6], [64, 16, 6], [32, 16, 6]]},
#     'activations': {'values': ['relu', 'tanh', 'sigmoid']},
#     'lr': {'values': [0.001, 0.01, 0.1]},
#     'batch_size': {'values': [32, 64, 128]}, 
#     'optimizer': {'values': ['sgd', 'bgd', 'mini_bgd']}
# }
# }

# data = DataLoader(RawDataDIR, "WineQT.csv")
# print(data.shape)
# print(data.describe())

# temp = data.drop(columns = ['quality'])
# labels = [temp.columns[i] for i in range(0, 12)]
# fig = px.scatter_matrix(data, dimensions = labels, color = 'quality', labels = {label: label for label in labels})
# fig.update_traces(diagonal_visible = True, showupperhalf = False)
# fig.update_layout(height = 1700, width = 1700, title_text="Pair Plot of Wine Features by Quality")
# fig.show()

# data = DataPreprocess(data, isMulti = False)
# print(data.describe())

# X = data.drop(columns = ['quality']).to_numpy()
# y = data['quality']
# num_classes = len(np.unique(y))
# labels = np.unique(y)-3
# Y = np.zeros((y.shape[0], num_classes))
# for i in range(y.shape[0]):
#     Y[i, y.iloc[i]-3] = 1

# indices = np.arange(0, X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# Y = Y[indices]

# X_train = X[:int(0.8*X.shape[0])]
# Y_train = Y[:int(0.8*Y.shape[0])]
# X_test = X[int(0.8*X.shape[0]):]
# Y_test = Y[int(0.8*Y.shape[0]):]

# def run_sweep():
#     sweep_agent_manager('Single_label_HPT', 'mlp_single', labels, X_train, X_test, Y_train, Y_test)

# sweep_id = wandb.sweep(sweep=config_sweep, project = 'Single_label_HPT')
# wandb.agent(sweep_id = sweep_id, 
#             function = run_sweep, 
#             count = 15)

# layers = [64, 32, 6]
# activations = 'relu'
# hyperparams = {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 256, 'optimizer': 'sgd', 'grad_verify': False, 
#                'labels': labels, 'layers': layers, 'activations': activations, 'wb': False}
# model = MultiLayerPerceptron_SingleClass(hyperparams)
# model.add()
# model.fit(X_train, Y_train, grad_verify = False)
# model.plot()
# print(model.evaluate(X_test, Y_test))
# wandb.finish()


################################### Multi label classification ###################################
wandb.login()

config_sweep = {
'method': 'bayes',
'name': 'Hyperparameter tuning: Multi label classification', 
'metric': {
    'goal': 'maximize',
    'name': 'soft accuracy'
}, 
'parameters': {
    'epochs': {'values': [50, 100, 500, 1000]},
    'layers': {'values': [[8], [16, 8], [64, 32, 16, 8], [20, 10, 8], [64, 32, 8], [64, 16, 8], [32, 16, 8]]},
    'activations': {'values': ['relu', 'tanh', 'sigmoid']},
    'lr': {'values': [0.0001, 0.001, 0.01, 0.1]},
    'batch_size': {'values': [32, 64, 256]}, 
    'optimizer': {'values': ['sgd', 'bgd', 'mini_bgd']},
    'thresh': {'values': [0.3, 0.5, 0.7, 0.9]}
}
}

data = DataLoader(RawDataDIR, "advertisement.csv")
print(data.shape)
print(data.describe())

encodings_gender = {'Male': 0, 'Female': 1}
data['gender'] = data['gender'].replace(encodings_gender)
data['married'] = data['married'].astype(int)

string_features = ['education', 'city', 'occupation', 'most bought item']
data = Word2Num(data, string_features)

scaler = StandardScaler()
temp = data['labels']
data = data.drop(columns = ['labels'])
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
data = pd.concat([data, temp], axis = 1)

X = data.drop(columns = ['labels']).to_numpy()
y = data['labels']
labels = []
for i in y:
    temp = i.split(' ')
    for j in temp:
        labels.append(j)
labels = np.unique(labels)
num_classes = len(labels)
Y = np.zeros((y.shape[0], num_classes))
for i in range(y.shape[0]):
    temp = y.iloc[i].split(' ')
    for j in temp:
        Y[i, np.where(labels == j)] = 1

indices = np.arange(0, X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train = X[:int(0.8*X.shape[0])]
Y_train = Y[:int(0.8*Y.shape[0])]
X_test = X[int(0.8*X.shape[0]):]
Y_test = Y[int(0.8*Y.shape[0]):]

def run_sweep():
    sweep_agent_manager('Multi_label_HPT_optimSA', 'mlp_multi', labels, X_train, X_test, Y_train, Y_test)

sweep_id = wandb.sweep(sweep=config_sweep, project = 'Multi_label_HPT_optimSA')
wandb.agent(sweep_id = sweep_id, 
            function = run_sweep, 
            count = 20)

# layers = [64, 32, 16, 8]
# activations = ['relu', 'relu', 'relu', 'sigmoid']
# hyperparams = {'learning_rate': 0.0001, 'epochs': 100, 'batch_size': 256, 'optimizer': 'sgd', 'num_classes': 8}
# model = createMLP(layers, activations, hyperparams, False)
# runMLP(model, data, grad_verify = False, isMulti = True)
# wandb.finish()

# import numpy as np
# import matplotlib.pyplot as plt

# class MLP_MultiLabel:
#     def __init__(self, input_size, output_size, hidden_layers, learning_rate=0.01, epochs=100, batch_size=32, activation='relu'):
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_layers = hidden_layers
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.activation = activation
#         self.weights = []
#         self.biases = []
#         self.history = {"loss": [], "accuracy": [], "f1_score": []}  # To track metrics
#         self.initialize_weights()

#     def initialize_weights(self):
#         # Initialize weights and biases for each layer
#         layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
#         for i in range(len(layer_sizes) - 1):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
#             self.biases.append(np.zeros((1, layer_sizes[i+1])))

#     def activation_function(self, x, derivative=False):
#         if self.activation == 'relu':
#             if derivative:
#                 return np.where(x > 0, 1, 0)
#             return np.maximum(0, x)
#         elif self.activation == 'sigmoid':
#             if derivative:
#                 sig = 1 / (1 + np.exp(-x))
#                 return sig * (1 - sig)
#             return 1 / (1 + np.exp(-x))
#         elif self.activation == 'tanh':
#             if derivative:
#                 return 1 - np.tanh(x)**2
#             return np.tanh(x)

#     def sigmoid(self, x, derivative=False):
#         if derivative:
#             sig = 1 / (1 + np.exp(-x))
#             return sig * (1 - sig)
#         return 1 / (1 + np.exp(-x))

#     def forward_propagation(self, X):
#         self.z_values = []
#         self.a_values = [X]
#         for i in range(len(self.weights) - 1):
#             z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
#             a = self.activation_function(z)
#             self.z_values.append(z)
#             self.a_values.append(a)
#         # Output layer with sigmoid activation (for multilabel classification)
#         z = np.dot(self.a_values[-1], self.weights[-1]) + self.biases[-1]
#         a = self.sigmoid(z)
#         self.z_values.append(z)
#         self.a_values.append(a)
#         return a

#     def back_propagation(self, X, Y):
#         m = X.shape[0]
#         dz = self.a_values[-1] - Y
#         dw = np.dot(self.a_values[-2].T, dz) / m
#         db = np.sum(dz, axis=0, keepdims=True) / m

#         gradients_w = [dw]
#         gradients_b = [db]

#         # Backpropagation for hidden layers
#         for i in reversed(range(len(self.weights) - 1)):
#             dz = np.dot(dz, self.weights[i+1].T) * self.activation_function(self.z_values[i], derivative=True)
#             dw = np.dot(self.a_values[i].T, dz) / m
#             db = np.sum(dz, axis=0, keepdims=True) / m
#             gradients_w.insert(0, dw)
#             gradients_b.insert(0, db)

#         return gradients_w, gradients_b

#     def update_weights(self, gradients_w, gradients_b):
#         for i in range(len(self.weights)):
#             self.weights[i] -= self.learning_rate * gradients_w[i]
#             self.biases[i] -= self.learning_rate * gradients_b[i]

#     def binary_cross_entropy_loss(self, Y, Y_pred):
#         m = Y.shape[0]
#         return -np.sum(Y * np.log(Y_pred + 1e-8) + (1 - Y) * np.log(1 - Y_pred + 1e-8)) / m

#     def predict(self, X):
#         Y_pred = self.forward_propagation(X)
#         return (Y_pred > 0.9).astype(int)

#     def hamming_loss(self, Y_true, Y_pred):
#         return np.mean(np.not_equal(Y_true, Y_pred))

#     def f1_score(self, Y_true, Y_pred):
#         precision = np.sum((Y_pred == 1) & (Y_true == 1), axis=0) / np.sum(Y_pred == 1, axis=0)
#         recall = np.sum((Y_pred == 1) & (Y_true == 1), axis=0) / np.sum(Y_true == 1, axis=0)
#         f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
#         return np.nanmean(f1)  # Handle division by zero

#     def accuracy(self, Y_true, Y_pred):
#         return np.mean(Y_pred == Y_true)

#     def fit(self, X_train, Y_train):
#         for epoch in range(self.epochs):
#             permutation = np.random.permutation(X_train.shape[0])
#             X_train = X_train[permutation]
#             Y_train = Y_train[permutation]

#             for i in range(0, X_train.shape[0], self.batch_size):
#                 X_batch = X_train[i:i+self.batch_size]
#                 Y_batch = Y_train[i:i+self.batch_size]
#                 Y_pred = self.forward_propagation(X_batch)
#                 gradients_w, gradients_b = self.back_propagation(X_batch, Y_batch)
#                 self.update_weights(gradients_w, gradients_b)

#             # Evaluate metrics after each epoch
#             Y_pred_train = self.forward_propagation(X_train)
#             loss = self.binary_cross_entropy_loss(Y_train, Y_pred_train)
#             accuracy = self.accuracy(Y_train, (Y_pred_train > 0.7).astype(int))
#             f1 = self.f1_score(Y_train, (Y_pred_train > 0.7).astype(int))
            
#             # Store metrics
#             self.history['loss'].append(loss)
#             self.history['accuracy'].append(accuracy)
#             self.history['f1_score'].append(f1)

#             if (epoch + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

#     def evaluate(self, X_test, Y_test):
#         Y_pred = self.predict(X_test)
#         accuracy = self.accuracy(Y_test, Y_pred)
#         hamming = self.hamming_loss(Y_test, Y_pred)
#         f1 = self.f1_score(Y_test, Y_pred)
#         print(f"Accuracy: {accuracy * 100:.2f}%")
#         print(f"Hamming Loss: {hamming:.4f}")
#         print(f"F1 Score: {f1:.4f}")

#     def plot_metrics(self):
#         epochs = range(1, self.epochs + 1)

#         plt.figure(figsize=(14, 6))

#         # Plot Loss
#         plt.subplot(1, 3, 1)
#         plt.plot(epochs, self.history['loss'], label='Loss')
#         plt.title('Loss over Epochs')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()

#         # Plot Accuracy
#         plt.subplot(1, 3, 2)
#         plt.plot(epochs, self.history['accuracy'], label='Accuracy', color='g')
#         plt.title('Accuracy over Epochs')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         # Plot F1 Score
#         plt.subplot(1, 3, 3)
#         plt.plot(epochs, self.history['f1_score'], label='F1 Score', color='r')
#         plt.title('F1 Score over Epochs')
#         plt.xlabel('Epochs')
#         plt.ylabel('F1 Score')
#         plt.legend()

#         plt.tight_layout()
#         plt.show()

# # Usage Example
# input_size = 10  # Example input size
# output_size = 8  # Example number of labels (multilabel)
# hidden_layers = [64, 32]  # Example hidden layer configuration

# mlp_model = MLP_MultiLabel(input_size, output_size, hidden_layers, learning_rate=0.01, epochs=100, batch_size=32, activation='relu')



# X = data.drop(columns = ['labels'])
# y = data['labels']
# labels = []
# for i in y:
#     temp = i.split(' ')
#     for j in temp:
#         labels.append(j)
# labels = np.unique(labels)
# num_classes = len(labels)
# Y = np.zeros((y.shape[0], num_classes))
# for i in range(y.shape[0]):
#     temp = y.iloc[i].split(' ')
#     for j in temp:
#         Y[i, np.where(labels == j)] = 1

# # X_train and Y_train should be your training data
# mlp_model.fit(X.to_numpy(), Y)

# # Evaluate on test data
# mlp_model.evaluate(X.to_numpy(), Y)

# mlp_model.plot_metrics()

################################### Regression ###################################
# data = DataLoader(RawDataDIR, "HousingData.csv")
# print(data.shape)
# print(data.describe())

# data = data.fillna(data.median())

# scaler = StandardScaler()
# data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
# print(data.describe())

# toy_data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]])
# toy_data = pd.DataFrame(toy_data, columns = ['A', 'B', 'C', 'D', 'E'])
# print(toy_data.describe())

# toy_data = toy_data - np.mean(toy_data, axis = 0)
# toy_data = toy_data / np.std(toy_data, axis = 0)
# # print(toy_data)
# X = toy_data.drop(columns = ['E']).to_numpy()
# y = toy_data['E'].to_numpy().reshape(-1, 1)

# X = data.drop(columns = ['MEDV']).to_numpy()
# y = data['MEDV'].to_numpy().reshape(-1, 1)
# layers = [64, 32, 1]
# activations = ['relu', 'relu', 'linear']
# hyperparams = {'learning_rate': 0.01, 'epochs': 500, 'batch_size': 16, 'optimizer': 'sgd', 'loss': 'mse'}
# model = MultiLayerPerceptron_Regression()
# for i in range(len(layers)):
#     layer = Layer(layers[i], activations[i])
#     model.add(layer)
# model.setHyperParams(hyperparams)
# model.fit(X, y, grad_verify = False)
# #plot predicted and true values
# plt.scatter(y, model._y_pred.flatten())
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('True vs Predicted Values')
# plt.show()
# model.plot()
# # wandb.finish()

################################### Autoencoder ###################################
# data = DataLoader(RawDataDIR, "HousingData.csv")
# print(data.shape)
# print(data.describe())

# data = data.fillna(data.median())

# scaler = StandardScaler()
# data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

# X = data.drop(columns = ['MEDV']).to_numpy()
# y = data['MEDV'].to_numpy().reshape(-1, 1)
# layers = [64, 32, 1]
# activations = ['relu', 'relu', 'linear']
# layers = [7, 5, 3, 5, 7, 14]
# activations = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'linear']
# hyperparams = {'learning_rate': 0.001, 'epochs': 500, 'batch_size': 16, 'optimizer': 'sgd', 'loss': 'mse'}
# model = MutliLayerPerceptron_Regression()
# for i in range(len(layers)):
#     layer = Layer(layers[i], activations[i])
#     model.add(layer)
# model.setHyperParams(hyperparams)
# model.fit(X, y, grad_verify = False)
# model.plot()
# wandb.finish()

