import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

import os
import sys 
import shutil
import time

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a5.py')))
CurrDIR = os.path.dirname(os.path.abspath('a5.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from sklearn.preprocessing import StandardScaler
import librosa

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/5/")

def sample_points_in_gaussian(n_points, mean, cov):
    """
    Sample n_points from a Gaussian distribution with given mean and covariance.

    Parameters:
        n_points (int): Number of points to sample
        mean (ndarray): Mean of the Gaussian distribution
        cov (ndarray): Covariance matrix of the Gaussian distribution
    """
    return np.random.multivariate_normal(mean, cov, n_points)

def sample_points_in_circle(n_points, radius, center=(0, 0)):
    """
    Sample n_points uniformly within a circle of given radius and center.

    Parameters:
        n_points (int): Number of points to sample
        radius (float): Radius of the circle
        center (tuple): Center of the circle
    """
    r = radius * np.sqrt(np.random.rand(n_points))
    theta = 2 * np.pi * np.random.rand(n_points)

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    
    return np.column_stack((x, y))

def getData(source: str, mean_small, mean_large, cov_small, cov_large):
    if source == 'circle':
        n_large = 3000
        large_circle = sample_points_in_circle(n_large, cov_large, mean_large)

        n_small = 500
        small_circle = sample_points_in_circle(n_small, cov_small, mean_small)
        
        data = np.concatenate([large_circle, small_circle])
    else:
        n_large = 3000
        large_circle = sample_points_in_gaussian(n_large, mean_large, cov_large)

        n_small = 500
        small_circle = sample_points_in_gaussian(n_small, mean_small, cov_small)
        
        data = np.concatenate([large_circle, small_circle])

    np.random.shuffle(data)
    pd.DataFrame(data, columns=['x', 'y']).to_csv(os.path.join(RawDataDIR, "data_a5.csv"), index=False)
        
    return data

def storeDataHMM(src: str, target: str):
    for root, _, file in os.walk(src):
        for f in file:
            if f.endswith('.wav'):
                digit = f.split('_')[0]
                shutil.move(os.path.join(root, f), os.path.join(target, digit))
    print(f"Data stored in {target}.")
    return None                

# storeDataHMM(os.path.join(RawDataDIR, 'recordings'), os.path.join(RawDataDIR, 'audio_mnist'))

################################### KDE ########################################
# from models.kde.kde import KDE

# data = getData('circle', [1, 1], [0, 0], 0.3, 2.25)

# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# type = 'triangular'
# kde = KDE(data)
# kde.buildKernel(type)
# kde.fit(type, 0.5)
# kde.plot(data, type)
# kde.plotKDE(data, type)


################################### HMM ########################################
# from models.hmm.hmm import HMM

# true_labels = []
# pred_labels = []
# mfccs_digit_pairs = {} 
# digits_likelihood = {str(i): [] for i in range(10)}
# hmm = HMM(n_states=10, n_epochs=100)
# for root, dir, files in os.walk(os.path.join(RawDataDIR, 'audio_mnist')):
#     mfccs = np.zeros((1, 13))
#     if (len(files) == 0):
#         continue
#     for f in files:
#         if f.endswith('.wav'):
#             true_labels.append(f.split('_')[0])
#             data, sr = librosa.load(os.path.join(root, f))
#             mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
#             mfccs = np.vstack((mfccs, mfcc.T))
#     mfccs = mfccs[1:]
#     mfccs_digit_pairs[root[-1]] = mfccs
#     print(mfccs.shape)
#     hmm.buildHMM(mfccs)
#     hmm.fitHMM(os.path.join('./', 'saved_models'), f'hmm_{root[-1]}.pkl')
#     print(f"Model for digit {root[-1]} trained successfully.")

# correct = 0
# total = 0
# for root, dir, files in os.walk(os.path.join(RawDataDIR, 'audio_mnist')):
#     for f in files:
#         if f.endswith('.wav'):
#             data, sr = librosa.load(os.path.join(root, f))
#             mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
#             _, pred = hmm.predictHMM(mfcc.T, os.path.join('./', 'saved_models'), 
#                                                      digits_likelihood, f.split('_')[0])
#             if (pred == f.split('_')[0]):
#                 correct += 1
#             total += 1

# print(f"Accuracy: {correct*100/total}")


