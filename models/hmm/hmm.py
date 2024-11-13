import numpy as np
from plotly import express as px
from plotly import subplots as sp
from plotly import graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from scipy.stats import multivariate_normal

from hmmlearn import hmm

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('kde.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

