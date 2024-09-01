import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys 
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import time

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a2.py')))
CurrDIR = os.path.dirname(os.path.abspath('a2.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/")