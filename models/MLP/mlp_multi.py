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

class MultiLayerPerceptron_MultiClass(object):
    ''' 
    MultiLayerPerceptron_MultiClass class for creating a neural network model for multi class 
    classification.
    '''
    def __init__(self) -> None:
        ''' 
        Initializes the MultiLayerPerceptron_MultiClass class.
        '''
        return None