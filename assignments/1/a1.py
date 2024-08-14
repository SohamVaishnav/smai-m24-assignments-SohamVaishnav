import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# from ...models.knn import knn
# from ...data.external import *

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a1.py')))
UserDIR = os.path.dirname(AssignDIR)
RawDataDIR = os.path.join(UserDIR, "./data/interim/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/")

assert os.path.exists(RawDataDIR), f"no!"
