import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys 
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import time

from sklearn.neighbors import KNeighborsClassifier

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('a1.py')))
CurrDIR = os.path.dirname(os.path.abspath('a1.py'))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

RawDataDIR = os.path.join(UserDIR, "./data/external/")
PreProcessDIR = os.path.join(UserDIR, "./data/interim/")
KNN_PreProcessDIR = os.path.join(PreProcessDIR, 'spotify_KNN/')

def DataLoader(DataDIR: str, datafilename: str, model: str):
    ''' 
    Loads the data from the directory. \n
    Note: Currently this function only deals with reading .csv files.

    Arguments:
    DataDIR = string denoting the path to the directory containing data.
    datafilename = string denoting the name of the data file that needs to be read from DataDIR
    model = the model that the data is being loaded for.
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)
    assert os.path.exists(data_path), f"{data_path} path is invalid!"

    if datafilename.endswith('.csv'):
        if (model == 'KNN'):
            data = pd.read_csv(data_path, sep = ',', index_col = 0)
            print("Data has been read.")
            print("Data size: (rows, cols)", data.shape)
            return data
        elif (model == 'LinReg'):
            x, y = np.loadtxt(data_path, delimiter = ',', dtype = np.float64, unpack = True, skiprows = 1)
            print("Data has been unpacked and loaded.")
            print("X: ", x.shape)
            print("Y: ", y.shape)
            return x, y

def DataWriter(DataDIR: str, datafilename: str, extension: str, data: pd.DataFrame) -> str:
    '''
    Writes data in the desired format (extension) at the given location having the desired name.\n
    Note: The function currently handles only .csv files.

    Arguments:
    DataDIR = the directory where data will be written into
    datafilename = the name of the file of data which will be written
    extension = the format of storage (e.g: .csv)
    '''
    assert os.path.exists(DataDIR), f"{DataDIR} path is invalid!"
    data_path = os.path.join(DataDIR, datafilename)
    data.to_csv(data_path, index = True)
    
    return data_path

def DataRefiner(dataset: pd.DataFrame,  model: str) -> pd.DataFrame:
    ''' 
    Performs preprocessing on the data like removing null and NaN, and duplicate datapoints as well.

    Arguments:
    dataset = the dataset that the user is dealing with.
    model = the kind of model that the data is being preprocessed for.
    '''
    if (model == 'KNN'):
        dataset = dataset.drop_duplicates(subset = ['track_id'], keep = 'first')
        dataset = dataset.dropna(axis = 0)
        dataset['duration_ms'] = dataset['duration_ms']/(60*1000)
        dataset = dataset.rename(columns = {'duration_ms':'duration_min'})
        print("Final shape of the Data: ", dataset.shape)
    return dataset

def Word2Num(dataset: pd.DataFrame, string_features, model: str) -> pd.DataFrame:
    ''' 
    Converts the string entries into some number which can then be used for classification. 
    
    Arguments:
    dataset = the original data that the model is dealing with
    string_features = the columns from the dataset that contain string entries
    model = denoted which model is the data being used for.
    '''
    for i in string_features:
        words, count = np.unique(dataset[i], return_counts = True)
        nums = count/np.max(count)
        encodings = {value: nums[i] for i, value in enumerate(words)}
        dataset[i] = dataset[i].replace(encodings)
    return dataset

################################### KNearest Neighbors ####################################
from models.knn.knn import KNN
    
model = KNN()
isValid = True
data = DataLoader(KNN_PreProcessDIR, 'spotify_word2num.csv', 'KNN')
data = data.drop(columns = ['key', 'mode', 'time_signature', 'duration_min'], axis = 1)
# data = model.DataRefiner(data)
string_features = ['track_id', 'artists', 'album_name', 'track_name', 'explicit']
# data = Word2Num(data, string_features, 'KNN')
# data = model.DataNormaliser(data)
train_set, valid_set, test_set = model.DataSplitter(0.8, isValid, 0.1, data, 'track_genre')
print("Training set: ", train_set.shape)
print("Testing set: ", test_set.shape)
print("Validation set: ", valid_set.shape)

y_train = train_set['track_genre']
X_train = train_set.drop(columns = ['track_genre'], axis = 1)

y_test = test_set['track_genre']
X_test = test_set.drop(columns = ['track_genre'], axis = 1)

y_valid = valid_set['track_genre']
X_valid = valid_set.drop(columns = ['track_genre'], axis = 1)

model.fit(X_train, y_train)

#################################### TASK 3 testing for different values
# k = [1, 5, 10, 15, 35, 50, 65, 80, 100, 150, 200]
# dist_metrics = ['cosine']
# results = DataLoader(CurrDIR, 'results_task3.csv', 'KNN')
# results = pd.DataFrame()

# start = time.time()
# idx = 0
# for dm in dist_metrics:
#     model.SetDistMetric(dm)
#     model.FindDistances(X_valid)
#     for i in k:
#         print("Number of Neighbors: ", i, "Distance Metric: ", dm)
#         model.SetNumNeighbors(i)
#         y_pred, acc, prec, recall, f1 = model.predict(X_valid, y_valid)
#         curr = pd.DataFrame([[dm, i, acc, prec[0], prec[1], recall[0], recall[1], f1[0], f1[1]]],
#                             index = [idx],
#                             columns = ['Dist_Metric', 'K', 'Accuracy', 'Precision_macro', 'Precision_micro', 'Recall_macro', 'Recall_micro', 'f1_macro', 'f1_micro'])
#         results = pd.concat([results, curr], axis = 0)
#         idx += 1

# print("Time taken: ", time.time()-start)
# DataWriter(CurrDIR, 'results_task3.csv', '.csv', results)

# start = time.time()
# best_k = 15
# best_dm = 'l2'
# model.SetDistMetric(best_dm)
# model.FindDistances(X_valid)
# model.SetNumNeighbors(best_k)
# print(model.predict(X_valid, y_valid)[1])
# print(time.time() - start)

####################################### Task 4
# exec_time_og = []
# exec_time_bm = []
# exec_time_mo = []
# exec_time_skl = []
# for i in [0.8, 0.6, 0.4, 0.2]:
#     train_set, valid_set, test_set = model.DataSplitter(i, isValid, 0.1, data, 'track_genre')
#     y_train = train_set['track_genre']
#     X_train = train_set.drop(columns = ['track_genre'], axis = 1)

#     y_test = test_set['track_genre']
#     X_test = test_set.drop(columns = ['track_genre'], axis = 1)

#     y_valid = valid_set['track_genre']
#     X_valid = valid_set.drop(columns = ['track_genre'], axis = 1)

#     start = time.time()
#     model.fit(X_train, y_train)
#     model.SetDistMetric('l1')
#     model.FindDistances(X_valid, 'initial')
#     model.SetNumNeighbors(1)
#     model.predict(X_valid, y_valid)
#     exec_time_og.append(time.time() - start)

#     start = time.time()
#     model.fit(X_train, y_train)
#     model.SetDistMetric('l1')
#     model.FindDistances(X_valid, 'initial')
#     model.SetNumNeighbors(15)
#     model.predict(X_valid, y_valid)
#     exec_time_bm.append(time.time() - start)

#     start = time.time()
#     model.fit(X_train, y_train)
#     model.SetDistMetric('l1')
#     model.FindDistances(X_valid, 'optimised')
#     model.SetNumNeighbors(15)
#     model.predict(X_valid, y_valid)
#     exec_time_mo.append(time.time() - start)

#     start = time.time()
#     model = KNeighborsClassifier(n_neighbors = 15, metric = 'l1')
#     model.fit(X_train, y_train)
#     model.predict(X_valid)
#     exec_time_skl.append(time.time() - start)

# fig1 = go.Line(x = [0.8, 0.6, 0.4, 0.2], y = exec_time_og, name = 'Original')
# fig2 = go.Line(x = [0.8, 0.6, 0.4, 0.2], y = exec_time_bm, name = 'Best Model')
# fig3 = go.Line(x = [0.8, 0.6, 0.4, 0.2], y = exec_time_mo, name = 'Model Optimised')
# fig4 = go.Line(x = [0.8, 0.6, 0.4, 0.2], y = exec_time_skl, name = 'Sklearn')
# fig = sp.make_subplots()
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.add_trace(fig3)
# fig.add_trace(fig4)
# fig.update_layout(title_text="Execution Time for KNN")
# fig.update_xaxes(title_text="Training Set Size")
# fig.update_yaxes(title_text="Execution Time")
# fig.show()

################################### Task 5
train_set = DataLoader(KNN_PreProcessDIR, '../spotify_2_KNN/train_w2n.csv', 'KNN')
valid_set = DataLoader(KNN_PreProcessDIR, '../spotify_2_KNN/valid_w2n.csv', 'KNN')
test_set = DataLoader(KNN_PreProcessDIR, '../spotify_2_KNN/test_w2n.csv', 'KNN')

y_train = train_set['track_genre']
X_train = train_set.drop(columns = ['duration_min', 'mode', 'key', 'time_signature', 'track_genre'], axis = 1)

y_test = test_set['track_genre']
X_test = test_set.drop(columns = ['duration_min', 'mode', 'key', 'time_signature', 'track_genre'], axis = 1)

y_valid = valid_set['track_genre']
X_valid = valid_set.drop(columns = ['duration_min', 'mode', 'key', 'time_signature', 'track_genre'], axis = 1)

k = 15
dist_metric = 'l1'
model.fit(X_train.iloc[0:700000], y_train.iloc[0:700000])
model.SetDistMetric(dist_metric)
model.FindDistances(X_train, 'optimised')
model.SetNumNeighbors(k)
y_pred, acc, prec, recall, f1 = model.predict(X_valid.iloc[0:9000], y_valid.iloc[0:9000])
print("Validation Set: ")
print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", recall)
print("F1 Score: ", f1)

y_pred, acc, prec, recall, f1 = model.predict(X_test.iloc[0:9000], y_test.iloc[0:9000])
print("Test Set: ")
print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", recall)
print("F1 Score: ", f1)

# train_set = model.DataRefiner(train_set)
# valid_set = model.DataRefiner(valid_set)
# test_set = model.DataRefiner(test_set)

# string_features = ['artists', 'album_name', 'track_name', 'explicit']
# train_set = train_set.drop(columns = ['track_id']) 
# train_set = Word2Num(train_set, string_features, 'KNN')
# train_set = model.DataNormaliser(train_set)
# DataWriter(KNN_PreProcessDIR, '../spotify_2_KNN/train_w2n.csv', '.csv', train_set)

# valid_set = Word2Num(valid_set, string_features, 'KNN')
# valid_set = valid_set.drop(columns = ['track_id'])
# valid_set = Word2Num(valid_set, string_features, 'KNN')
# valid_set = model.DataNormaliser(valid_set)
# DataWriter(KNN_PreProcessDIR, '../spotify_2_KNN/valid_w2n.csv', '.csv', valid_set)

# test_set = Word2Num(test_set, string_features, 'KNN')
# test_set = test_set.drop(columns = ['track_id'])
# test_set = Word2Num(test_set, string_features, 'KNN')
# test_set = model.DataNormaliser(test_set)
# DataWriter(KNN_PreProcessDIR, '../spotify_2_KNN/test_w2n.csv', '.csv', test_set)


################################ Linear Regression #################################
# from models.linear_regression.linear_regression import LinearRegression

# x, y = DataLoader(RawDataDIR, 'linreg.csv', 'LinReg')
# model = LinearRegression()
# print(np.std(X_train), np.std(y_train))
# print(np.std(X_valid), np.std(y_valid))
# print(np.std(X_test), np.std(y_test))

# print(np.var(X_train), np.var(y_train))
# print(np.var(X_valid), np.var(y_valid))
# print(np.var(X_test), np.var(y_test))

########## Degree = 1
# degree = 1
# lr = 0.0001
# model.SetDegree(degree)
# x_poly = model.Transform2Poly(x)
# X_train, y_train, X_valid, y_valid, X_test, y_test = model.DataSplitter(0.8, True, 0.1, x_poly, y)
# model.SetLearningRate(lr)
# beta, mse = model.fit(X_train, y_train, 'gd',  1000)
# print("beta: ", beta)
# print("train_mse: ", mse)

# y_pred, mse = model.predict(X_test, y_test)
# print("test_mse: ", mse)

# y_pred = y_pred.reshape((y_pred.shape[0],))

# fig1 = go.Scatter(x = X_test, y = y_test, mode = 'markers', name = 'Original Data')
# fig2 = go.Scatter(x = X_test, y = y_pred, mode = 'markers', name = 'Line of Best Fit')
# fig = sp.make_subplots()
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.update_layout(title_text="Linear Regression of Degree 1")
# fig.show()

########## Degree > 1
# fig = sp.make_subplots(rows=2, cols=4)
# l, k = 1, 1
# for degree in [2, 3, 4, 5, 6, 7, 10, 12]:
#     print(degree)
#     lr = 0.0001
#     model.SetDegree(degree)
#     x_poly = model.Transform2Poly(x)
#     X_train, y_train, X_valid, y_valid, X_test, y_test = model.DataSplitter(0.8, True, 0.1, x_poly, y)
#     model.SetLearningRate(lr)
#     beta, mse, y_pred = model.fit(X_train, y_train, 'gd',  1000)
#     print(np.var(y_pred), np.std(y_pred))
#     print("train_mse: ", mse)

#     y_pred, mse = model.predict(X_test, y_test)
#     print("test_mse: ", mse)
#     print(np.var(y_pred), np.std(y_pred))

#     y_pred = y_pred.reshape((y_pred.shape[0],))

#     fig1 = go.Scatter(x = X_test[:,0], y = y_test, mode = 'markers', name = 'Original Data', showlegend=False, marker = dict(color = 'lightskyblue'))
#     fig2 = go.Scatter(x = X_test[:,0], y = y_pred, mode = 'markers', name = 'Line of Best Fit', showlegend=False, marker = dict(color = 'salmon'))
#     fig.append_trace(fig1, row = l, col = k)
#     fig.append_trace(fig2, row = l, col = k)
#     fig.update_layout(title_text="Degree "+str(degree))
#     if (k == 4):
#         l += 1
#         k = 1
#     else:
#         k += 1
    

# fig.update_layout(title_text="Linear Regression of Degree > 1")
# fig.show()

################################ Regularisation ###################################
# from models.linear_regression.linear_regression import LinearRegression

# x, y = DataLoader(RawDataDIR, 'regularisation.csv', 'LinReg')

# for type in ['l1', 'l2']:
#     model = LinearRegression(lr=0.0001, l = 0)
#     fig = sp.make_subplots(rows=4, cols=5)
#     l, k = 1, 1
#     for degree in range(2, 21):
#         # lr = 0.0001
#         model.SetDegree(degree)
#         x_poly = model.Transform2Poly(x)
#         regularise = True
#         X_train, y_train, X_valid, y_valid, X_test, y_test = model.DataSplitter(0.8, True, 0.1, x_poly, y)

#         beta, mse, pred = model.fit(X_train, y_train, 'gd', 1000, regularise, type)
#         print(mse)
#         print(np.var(pred), np.std(pred))
#         y_pred, mse = model.predict(X_test, y_test)
#         print(mse)
#         print(np.var(y_pred), np.std(y_pred))

#         y_pred = y_pred.reshape((y_pred.shape[0],))
#         fig1 = go.Scatter(x = X_test[:,0], y = y_test, mode = 'markers', name = 'Original Data', showlegend=False, marker = dict(color = 'lightskyblue'))
#         fig2 = go.Scatter(x = X_test[:,0], y = y_pred, mode = 'markers', name = 'Line of Best Fit', showlegend=False, marker = dict(color = 'salmon'))
#         fig.append_trace(fig1, row = l, col = k)
#         fig.append_trace(fig2, row = l, col = k)
#         fig.update_layout(title_text="Degree "+str(degree))
#         if (k == 5):
#             l += 1
#             k = 1
#         else:
#             k += 1
#     fig.update_layout(title_text="Without Regularisation")
#     fig.show()

################################ VISUALISATION ###################################

#KNN

#genres
# fig1 = go.Histogram(x = data['track_genre'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# fig2 = go.Histogram(x = data['track_genre'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Genre Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#popularity
# fig1 = go.Histogram(x = data['popularity'], y = data['popularity'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['popularity'], y = data['popularity'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Popularity Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#explicit
# fig1 = go.Pie(values = data['explicit'].value_counts(), labels = ['Not Explicit', 'Explicit'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# fig2 = go.Pie(values = data['explicit'].value_counts(), labels = ['Not Explicit', 'Explicit'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Explicit Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#mode
# fig1 = go.Pie(values = data['mode'].value_counts(), labels = ['0', '1'], name = 'Mode Distribution')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Mode Distribution")
# fig.add_trace(fig1)
# fig.show()


#duration_ms
# fig1 = go.Histogram(x = data['duration_ms'], name = 'Before Preprocessing (in milliseconds)')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['duration_min'], name = 'After Preprocessing (in minutes)')
# fig = sp.make_subplots(rows=1, cols=2)
# fig.update_layout(title_text="Duration Distribution Before and After Preprocessing")
# fig.add_trace(fig1, row=1, col=1)
# fig.add_trace(fig2, row=1, col=2)
# fig.show()


#danceability
# fig1 = go.Histogram(x = data['danceability'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['danceability'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Danceability Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#energy
# fig1 = go.Histogram(x = data['energy'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['energy'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Energy Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#key
# fig1 = go.Histogram(x = data['key'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['key'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Key Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#loudness
# fig1 = go.Histogram(x = data['loudness'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['loudness'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Loudness Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#speechiness
# fig1 = go.Histogram(x = data['speechiness'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['speechiness'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Speechiness Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#acousticness
# fig1 = go.Histogram(x = data['acousticness'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['acousticness'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Acousticness Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()

# fig1 = go.Box(x = data['acousticness'], name = 'Before Preprocessing')
# fig3 = go.Violin(x = data['acousticness'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# model.DataNormaliser(data)
# fig2 = go.Box(x = data['acousticness'], name = 'After Preprocessing')
# fig = sp.make_subplots(rows=1, cols=2)
# fig.update_layout(title_text="Acousticness Distribution Before and After Preprocessing")
# fig.add_trace(fig1, row=1, col=1)
# fig.add_trace(fig2, row=1, col=1)
# fig.add_trace(fig3, row=1, col=2)
# fig.show()


#instrumentalness
# fig1 = go.Histogram(x = data['instrumentalness'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['instrumentalness'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Instrumentalness Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#liveness
# fig1 = go.Histogram(x = data['liveness'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['liveness'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Liveness Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#valence
# fig1 = go.Histogram(x = data['valence'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['valence'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Valence Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()

# fig1 = go.Box(x = data['valence'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Box(x = data['valence'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Valence Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#tempo
# fig1 = go.Histogram(x = data['tempo'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Histogram(x = data['tempo'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Tempo Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()

# fig1 = go.Box(x = data['tempo'], name = 'Before Preprocessing')
# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# fig2 = go.Box(x = data['tempo'], name = 'After Preprocessing')
# fig = sp.make_subplots()
# fig.update_layout(title_text="Tempo Distribution Before and After Preprocessing")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.show()


#time_signature
# fig1 = go.Pie(values = data['time_signature'].value_counts())
# fig = sp.make_subplots()
# fig.update_layout(title_text="Time Signature Distribution")
# fig.add_trace(fig1)
# fig.show()


#all combined
# fig = sp.make_subplots(rows=4, cols=3, subplot_titles = ('Popularity', 'Duration', 'Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo'))
# k, l = 1, 1
# labels = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
# for i in range(0, len(labels)):
#     fig1 = go.Histogram(x = data[labels[i]], showlegend=False, marker = dict(color = 'lightskyblue'))
#     fig.append_trace(fig1, row = k, col = l)
#     if (l == 3):
#         k += 1
#         l = 1
#     else:
#         l += 1

# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# k, l = 1, 1
# labels = ['popularity', 'duration_min', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
# for i in range(0, len(labels)):
#     fig2 = go.Histogram(x = data[labels[i]], showlegend=False, marker = dict(color = 'salmon'))
#     fig.append_trace(fig2, row = k, col = l)
#     if (l == 3):
#         k += 1
#         l = 1
#     else:
#         l += 1
# fig.update_layout(title_text="Feature Distribution Before and After Preprocessing")
# fig.update_layout(showlegend=True)
# fig.show()


#Pair Plot
# labels = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
# fig = px.scatter_matrix(data, dimensions = labels, color = 'track_genre', labels = {label: label for label in labels})
# fig.update_traces(diagonal_visible = True, showupperhalf = False)
# fig.update_layout(height = 1700, width = 1700, title_text="Pair Plot of Music Features by Genre")
# fig.show()

# data = model.DataRefiner(data)
# data = model.DataNormaliser(data)
# labels = ['popularity', 'duration_min', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
# fig = px.scatter_matrix(data, dimensions = labels, color = 'track_genre', labels = {label: label for label in labels})
# fig.update_traces(diagonal_visible = True, showupperhalf = False)
# fig.update_layout(height = 1700, width = 1700, title_text = "Pair Plot of Music Features by Genre after Preprocessing")
# fig.show()


# Pair Plot for Word2Num
# label1 = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'tempo']
# label2 = ['album_name', 'track_name', 'artists']
# fig = px.scatter_matrix(data, dimensions = label1 + label2, color = 'track_genre', labels = {label: label for label in label1 + label2})
# fig.update_traces(diagonal_visible = False, showupperhalf = False)
# fig.update_layout(height = 1700, width = 1700, title_text = "Pair Plot of Selected Features by Genre after Preprocessing and Word2Num")
# fig.show()


#Linear Regression

# Entire Data
# fig1 = go.Scatter(x = x, y = y, mode = 'markers', name = 'Original Data')
# fig = sp.make_subplots()
# fig.update_xaxes(title_text="X")
# fig.update_yaxes(title_text="Y")
# fig.add_trace(fig1)
# fig.update_layout(title_text="Original Data")
# fig.show()

# Splitted Data
# fig1 = go.Scatter(x = X_train, y = y_train, mode = 'markers', name = 'Training Data')
# fig2 = go.Scatter(x = X_valid, y = y_valid, mode = 'markers', name = 'Validation Data')
# fig3 = go.Scatter(x = X_test, y = y_test, mode = 'markers', name = 'Testing Data')
# fig = sp.make_subplots()
# fig.update_xaxes(title_text="X")
# fig.update_yaxes(title_text="Y")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.add_trace(fig3)
# fig.update_layout(title_text="Splitted Data for Linear Regression w/o Regularisation")
# fig.show()

#for Regularisation
# fig1 = go.Scatter(x = x, y = y, mode = 'markers', name = 'Original Data')
# fig = sp.make_subplots()
# fig.update_xaxes(title_text="X")
# fig.update_yaxes(title_text="Y")
# fig.add_trace(fig1)
# fig.update_layout(title_text="Original Data")
# fig.show()

# Splitted Data
# fig1 = go.Scatter(x = X_train[:,0], y = y_train, mode = 'markers', name = 'Training Data')
# fig2 = go.Scatter(x = X_valid[:,0], y = y_valid, mode = 'markers', name = 'Validation Data')
# fig3 = go.Scatter(x = X_test[:,0], y = y_test, mode = 'markers', name = 'Testing Data')
# fig = sp.make_subplots()
# fig.update_xaxes(title_text="X")
# fig.update_yaxes(title_text="Y")
# fig.add_trace(fig1)
# fig.add_trace(fig2)
# fig.add_trace(fig3)
# fig.update_layout(title_text="Splitted Data for Linear Regression with Regularisation")
# fig.show()




