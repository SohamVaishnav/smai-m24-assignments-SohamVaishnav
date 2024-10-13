import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import wandb

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('utils.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from models.MLP.mlp_regression import *
from models.AutoEncoders.auto_encoder import *
from models.MLP.mlp import *
from models.MLP.mlp_multi import *
from performance_measures.metricsMLP import *

def createModel(config):
    type = config['type']
    if type == 'regression':
        model = MutliLayerPerceptron_Regression(config)
        model.add()
        return model
    elif type == 'autoencoder':
        model = AutoEncoder(config)
        return model
    elif type == 'mlp_single':
        model = MultiLayerPerceptron_SingleClass(config)
        model.add()
        return model
    elif type == 'mlp_multi':
        print("mlp_multi")
        model = MultiLayerPerceptron_MultiClass(config)
        model.add()
        return model

def runModel(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train, X_valid, y_valid)
    # model.plot()

def evalModel(model, X, y):
    metrics = model.evaluate(X, y)
    return metrics

def sweep_agent_manager(project_name: str, model: str, X_train: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_valid: np.ndarray, y_test: np.ndarray, labels = None):
    ''' 
    This function is used to manage the sweep agent for hyperparameter tuning.
    
    Parameters:
        project_name = name of the project.
        model = model type.
        labels = list of labels.
        X_train = training data.
        X_test = testing data.
        y_train = training labels.
        y_test = testing labels.
    '''
    wandb.init(project=project_name, entity="soham-iiith")
    config = dict(wandb.config)

    hyperparams = {}
    if (model == 'mlp_single' or model == 'mlp_multi'):
        hyperparams = {'learning_rate': [], 'epochs': [], 'batch_size': [], 'optimizer': [], 'grad_verify': False, 
                       'labels': [], 'layers': [], 'activations': [], 'thresh': [], 'type': model, 'wb': True}
        hyperparams['learning_rate'] = config['lr']
        hyperparams['epochs'] = config['epochs']
        hyperparams['batch_size'] = config['batch_size']
        hyperparams['optimizer'] = config['optimizer']
        hyperparams['labels'] = labels
        hyperparams['layers'] = config['layers']
        hyperparams['activations'] = config['activations']
        hyperparams['thresh'] = config['thresh']

    elif (model == 'regression' or model == 'autoencoder'):
        hyperparams = {'learning_rate': [], 'epochs': [], 'batch_size': [], 'optimizer': [], 'grad_verify': False, 
                       'loss': [], 'layers': [], 'activations': [], 'type': model, 'wb': True}
        hyperparams['learning_rate'] = config['lr']
        hyperparams['epochs'] = config['epochs']
        hyperparams['batch_size'] = config['batch_size']
        hyperparams['loss'] = config['loss']
        hyperparams['layers'] = config['layers']
        hyperparams['activations'] = config['activations']
        hyperparams['optimizer'] = config['optimizer']
    
    run_name = f"{model}_ep{hyperparams['epochs']}_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}_ls{hyperparams['layers']}_acts{hyperparams['activations']}"    
    wandb.run.name = run_name

    model = createModel(hyperparams)
    runModel(model, X_train, y_train, X_valid, y_valid)
    #write the predictions to a csv file
    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred, columns=labels).to_csv(f"Task2_7_predictions.csv", index=False)
    print(evalModel(model, X_test, y_test))
    wandb.finish()
