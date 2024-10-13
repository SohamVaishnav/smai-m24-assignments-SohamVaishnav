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
from models.knn.knn import *
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
#     'name': 'val accuracy'
# }, 
# 'parameters': {
#     'epochs': {'values': [50, 100, 500, 1000]},
#     'layers': {'values': [[6], [15, 6], [64, 32, 6], [64, 16, 6], [32, 16, 6]]},
#     'activations': {'values': ['relu', 'tanh', 'sigmoid']},
#     'lr': {'values': [0.001, 0.01, 0.1]},
#     'batch_size': {'values': [32, 64, 128]}, 
#     'optimizer': {'values': ['sgd', 'bgd', 'mini_bgd']}, 
#     'thresh': {'values': [0.3, 0.5, 0.7, 0.9]}
# }
# }

# config_sweep = {
# 'method': 'grid',
# 'name': 'Hyperparameter tuning: Single label classification', 
# 'metric': {
#     'goal': 'maximize',
#     'name': 'val accuracy'
# }, 
# 'parameters': {
#     'epochs': {'values': [50]},
#     'layers': {'values': [[32, 16, 6]]},
#     'activations': {'values': ['relu']},
#     'lr': {'values': [0.01]},
#     'batch_size': {'values': [32, 64, 128, 256]}, 
#     'optimizer': {'values': ['mini_bgd']}, 
#     'thresh': {'values': [0.3]}
# }
# }

# data = DataLoader(RawDataDIR, "WineQT.csv")
# print(data.shape)
# print(data.describe())

# # temp = data.drop(columns = ['quality'])
# # labels = [temp.columns[i] for i in range(0, 12)]
# # fig = px.scatter_matrix(data, dimensions = labels, color = 'quality', labels = {label: label for label in labels})
# # fig.update_traces(diagonal_visible = True, showupperhalf = False)
# # fig.update_layout(height = 1700, width = 1700, title_text="Pair Plot of Wine Features by Quality")
# # fig.show()

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
# X_valid = X[int(0.8*X.shape[0]):int(0.9*X.shape[0])]
# Y_valid = Y[int(0.8*Y.shape[0]):int(0.9*Y.shape[0])]
# X_test = X[int(0.9*X.shape[0]):]
# Y_test = Y[int(0.9*Y.shape[0]):]

# def run_sweep():
#     sweep_agent_manager('Single_label_HPT_2.5.3', 'mlp_single', X_train, X_valid, X_test, Y_train, Y_valid, Y_test, labels)

# sweep_id = wandb.sweep(sweep=config_sweep, project = 'Single_label_HPT_2.5.3')
# wandb.agent(sweep_id = sweep_id, 
#             function = run_sweep, 
#             count = 4)

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
# wandb.login()

# config_sweep = {
# 'method': 'bayes',
# 'name': 'Hyperparameter tuning: Multi label classification', 
# 'metric': {
#     'goal': 'maximize',
#     'name': 'soft accuracy'
# }, 
# 'parameters': {
#     'epochs': {'values': [50, 100, 500, 1000]},
#     'layers': {'values': [[8], [16, 8], [64, 32, 16, 8], [20, 10, 8], [64, 32, 8], [64, 16, 8], [32, 16, 8]]},
#     'activations': {'values': ['relu', 'tanh', 'sigmoid']},
#     'lr': {'values': [0.0001, 0.001, 0.01, 0.1]},
#     'batch_size': {'values': [32, 64, 256]}, 
#     'optimizer': {'values': ['sgd', 'bgd', 'mini_bgd']},
#     'thresh': {'values': [0.3, 0.5, 0.7, 0.9]}
# }
# }

# config_sweep = {
# 'method': 'bayes',
# 'name': 'Hyperparameter tuning: Multi label classification', 
# 'metric': {
#     'goal': 'maximize',
#     'name': 'soft accuracy'
# }, 
# 'parameters': {
#     'epochs': {'values': [100]},
#     'layers': {'values': [[64, 32, 16, 8]]},
#     'activations': {'values': ['tanh']},
#     'lr': {'values': [0.01]},
#     'batch_size': {'values': [32, 64, 256]}, 
#     'optimizer': {'values': ['mini_bgd']},
#     'thresh': {'values': [0.3]}
# }
# }

# data = DataLoader(RawDataDIR, "advertisement.csv")
# print(data.shape)
# print(data.describe())

# fig = px.histogram(data, x = 'age', title = 'Age distribution of the people')
# fig.show()

# fig1 = go.Pie(values = data['married'].value_counts(), labels = ['True', 'False'], name = 'Marriage Status')
# fig2 = go.Pie(values = data['gender'].value_counts(), labels = ['Male', 'Female'], name = 'Gender')
# fig = sp.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
# fig.update_layout(title_text="Marriage and Gender Distributions")
# fig.add_trace(fig1, row = 1, col = 1)
# fig.add_trace(fig2, row = 1, col = 2)
# fig.show()

# fig = px.histogram(data, x = 'most bought item', title = 'Most bought item distribution')
# fig.show()

# fig = px.scatter(data, x = 'city', y = 'most bought item', color = 'most bought item', title = 'City distribution with respect to most bought item')
# fig.show()

# fig = px.scatter(data, x = 'city', y = 'gender', color = 'gender', title = 'City distribution with respect to Gender')
# fig.show()

# fig = px.scatter(data, x = 'most bought item', y = 'gender', color = 'most bought item', title = 'Items bought with respect to Gender')
# fig.show()

# fig = px.scatter(data, x = 'city', y = 'most bought item', color = 'education', title = 'Items bought with respect to city and education')
# fig.show()

# fig = px.scatter(data, x = 'city', y = 'most bought item', color = 'occupation', title = 'Items bought with respect to city and occupation')
# fig.show()

# encodings_gender = {'Male': 0, 'Female': 1}
# data['gender'] = data['gender'].replace(encodings_gender)
# data['married'] = data['married'].astype(int)

# string_features = ['education', 'city', 'occupation', 'most bought item']
# data = Word2Num(data, string_features)

# scaler = StandardScaler()
# temp = data['labels']
# data = data.drop(columns = ['labels'])
# data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
# data = pd.concat([data, temp], axis = 1)

# X = data.drop(columns = ['labels']).to_numpy()
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

# indices = np.arange(0, X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# Y = Y[indices]

# X_train = X[:int(0.8*X.shape[0])]
# Y_train = Y[:int(0.8*Y.shape[0])]
# X_valid = X[int(0.8*X.shape[0]):int(0.9*X.shape[0])]
# Y_valid = Y[int(0.8*Y.shape[0]):int(0.9*Y.shape[0])]
# X_test = X[int(0.9*X.shape[0]):]
# Y_test = Y[int(0.9*Y.shape[0]):]

# def run_sweep():
#     sweep_agent_manager('Multi_label_HPT_optimSA_trial3', 'mlp_multi', X_train, X_valid, X_test, Y_train, Y_valid, Y_test, labels)

# sweep_id = wandb.sweep(sweep=config_sweep, project = 'Multi_label_HPT_optimSA_trial3')
# wandb.agent(sweep_id = sweep_id, 
#             function = run_sweep, 
#             count = 20)

# layers = [64, 32, 16, 8]
# activations = ['relu', 'relu', 'relu', 'sigmoid']
# hyperparams = {'learning_rate': 0.0001, 'epochs': 100, 'batch_size': 256, 'optimizer': 'sgd', 'num_classes': 8}
# model = createMLP(layers, activations, hyperparams, False)
# runMLP(model, data, grad_verify = False, isMulti = True)
# wandb.finish()


################################### Regression ###################################
# wandb.login()

# config_sweep = {
# 'method': 'bayes',
# 'name': 'Hyperparameter tuning: Multi label classification', 
# 'metric': {
#     'goal': 'minimize',
#     'name': 'val loss'
# }, 
# 'parameters': {
#     'epochs': {'values': [50, 100, 500, 1000]},
#     'layers': {'values': [[1], [16, 1], [64, 32, 16, 1], [16, 8, 1], [64, 32, 1], [64, 16, 1], [32, 16, 1]]},
#     'activations': {'values': ['relu', 'tanh', 'sigmoid']},
#     'lr': {'values': [0.0001, 0.001, 0.01, 0.1]},
#     'batch_size': {'values': [32, 64, 256]}, 
#     'optimizer': {'values': ['sgd', 'bgd', 'mini_bgd']}, 
#     'loss': {'values': ['cross_entropy']}
# }
# }

# config_sweep = {
# 'method': 'bayes',
# 'name': 'Hyperparameter tuning: Multi label classification', 
# 'metric': {
#     'goal': 'minimize',
#     'name': 'val loss'
# }, 
# 'parameters': {
#     'epochs': {'values': [50]},
#     'layers': {'values': [[16, 8, 1]]},
#     'activations': {'values': ['relu']},
#     'lr': {'values': [0.01]},
#     'batch_size': {'values': [256]}, 
#     'optimizer': {'values': ['sgd']}, 
#     'loss': {'values': ['mse']}
# }
# }

# data = DataLoader(RawDataDIR, "diabetes.csv")
# print(data.shape)
# print(data.describe())

# print(data.isnull().sum())

# data = data.fillna(data.median())

# # fig = px.scatter_matrix(data, dimensions = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'], color = 'MEDV', labels = {'MEDV': 'Value in $1000s'})
# # fig.update_traces(diagonal_visible = True, showupperhalf = False)
# # fig.update_layout(height = 1700, width = 1700, title_text="Pair Plot of Housing Features by Median Value")
# # fig.show()

# scaler = StandardScaler()
# # data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
# # print(data.describe())


# X = data.drop(columns = ['Outcome'])
# X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
# X = X.to_numpy()
# y = data['Outcome'].to_numpy().reshape(-1, 1)

# indices = np.arange(0, X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# Y = y[indices]

# X_train = X[:int(0.8*X.shape[0])]
# Y_train = Y[:int(0.8*Y.shape[0])]
# X_valid = X[int(0.8*X.shape[0]):int(0.9*X.shape[0])]
# Y_valid = Y[int(0.8*Y.shape[0]):int(0.9*Y.shape[0])]
# X_test = X[int(0.9*X.shape[0]):]
# Y_test = Y[int(0.9*Y.shape[0]):]

# def run_sweep():
#     sweep_agent_manager(project_name='MLP_Regression_HPT_3.5', model = 'regression', X_train = X_train, X_valid = X_valid, X_test = X_test, y_train = Y_train, y_valid = Y_valid, y_test = Y_test)

# sweep_id = wandb.sweep(sweep=config_sweep, project = 'MLP_Regression_HPT_3.5')
# wandb.agent(sweep_id = sweep_id, 
#             function = run_sweep, 
#             count = 1)

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
# wandb.login()

config_sweep = {
'method': 'bayes',
'name': 'Hyperparameter tuning: Autoencoder', 
'metric': {
    'goal': 'minimize',
    'name': 'loss'
}, 
'parameters': {
    'epochs': {'values': [100]},
    'layers': {'values': [[17, 11, 17, 18]]},
    'activations': {'values': ['relu']},
    'lr': {'values': [0.001]},
    'batch_size': {'values': [256]}, 
    'optimizer': {'values': ['sgd']}, 
    'loss': {'values': ['mse']}
}
}

KNN_DIR = os.path.join(UserDIR, "./data/interim/1/spotify_KNN")
data = pd.read_csv(os.path.join(KNN_DIR, "spotify_word2num.csv"), index_col = 0)
print(data.shape)
print(data.describe())

hyperparams = {'learning_rate': [], 'epochs': [], 'batch_size': [], 'optimizer': [], 'grad_verify': False, 
                       'loss': [], 'layers': [], 'activations': [], 'type': 'autoencoder', 'wb': False}
hyperparams['learning_rate'] = config_sweep['parameters']['lr']['values'][0]
hyperparams['epochs'] = config_sweep['parameters']['epochs']['values'][0]
hyperparams['batch_size'] = config_sweep['parameters']['batch_size']['values'][0]
hyperparams['loss'] = config_sweep['parameters']['loss']['values'][0]
hyperparams['layers'] = config_sweep['parameters']['layers']['values'][0]
hyperparams['activations'] = config_sweep['parameters']['activations']['values'][0]
hyperparams['optimizer'] = config_sweep['parameters']['optimizer']['values'][0]

# temp = data.drop(columns = ['track_genre'])

# print(temp.to_numpy())

# model_2 = AutoEncoder(hyperparams)
# model_2.fit(temp.to_numpy(), temp.to_numpy(), temp.to_numpy(), temp.to_numpy())
# model_2.evaluate(temp.to_numpy(), temp.to_numpy())

# temp_latent = model_2.get_latent(temp.to_numpy())
# data_latent = pd.concat([pd.DataFrame(temp_latent), data['track_genre']], axis = 1)
# np.savetxt(os.path.join(PreProcessDIR, "data_latent.csv"), temp_latent, delimiter = ",")

data_latent = pd.read_csv(os.path.join(PreProcessDIR, "data_latent.csv"))
data_latent = data_latent.iloc[:, :4]
data_latent['track_genre'] = data['track_genre'].values

model_1 = KNN()
isValid = True
train_set, valid_set, test_set = model_1.DataSplitter(0.8, isValid, 0.1, data_latent, 'track_genre')
print("Training set: ", train_set.shape)
print("Testing set: ", test_set.shape)
print("Validation set: ", valid_set.shape)

y_train = train_set['track_genre']
X_train = train_set.drop(columns = ['track_genre'], axis = 1)

y_test = test_set['track_genre'] 
X_test = test_set.drop(columns = ['track_genre'], axis = 1)

y_valid = valid_set['track_genre']
X_valid = valid_set.drop(columns = ['track_genre'], axis = 1)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

# model_2 = AutoEncoder(hyperparams)
# model_2.fit(X_train, X_train, X_valid, X_valid)
# model_2.evaluate(X_test, X_test)

# X_train_latent = model_2.get_latent(X_train)
# X_valid_latent = model_2.get_latent(X_valid)
# X_test_latent = model_2.get_latent(X_test)

# print(X_train_latent.shape)
# print(X_valid_latent.shape)
# print(X_test_latent.shape)

# np.savetxt(os.path.join(PreProcessDIR, "X_train_latent.csv"), X_train_latent, delimiter = ",")
# np.savetxt(os.path.join(PreProcessDIR, "X_valid_latent.csv"), X_valid_latent, delimiter = ",")
# np.savetxt(os.path.join(PreProcessDIR, "X_test_latent.csv"), X_test_latent, delimiter = ",")

# def run_sweep():
#     sweep_agent_manager(project_name='AutoEncoder_HPT', model = 'autoencoder', X_train = X_train, X_test = X_valid, y_train = X_train, y_test = X_valid)

# sweep_id = wandb.sweep(sweep=config_sweep, project = 'AutoEncoder_HPT')
# wandb.agent(sweep_id = sweep_id, 
#             function = run_sweep, 
#             count = 5)

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

# X_train_latent = pd.read_csv(os.path.join(PreProcessDIR, "X_train_latent.csv"))
# X_valid_latent = pd.read_csv(os.path.join(PreProcessDIR, "X_valid_latent.csv"))
# X_test_latent = pd.read_csv(os.path.join(PreProcessDIR, "X_test_latent.csv"))

time_start = time.time()
model_1.fit(X_train, y_train)
model_1.SetDistMetric('l1')
model_1.SetNumNeighbors(15)
model_1.FindDistances(X_valid, 'optimised')
y_pred, acc, prec, recall, f1 = model_1.predict(X_valid, y_valid)
time_end = time.time()
print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Time taken: ", time_end - time_start)