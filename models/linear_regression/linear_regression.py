import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os

AssignDIR = os.path.dirname(os.path.dirname(os.path.abspath('linear_regression.py')))
UserDIR = os.path.dirname(AssignDIR)

sys.path.append(UserDIR)

from performance_measures.confusion_matrix import Confusion_Matrix, Measures

class LinearRegression:
    ''' 
    This class contains all the functions related to Linear Regression algorithm. To
    run the algorithm over a dataset, create a model in the main code by calling the class and running 
    the 'train', 'eval' and 'test' functions as and when required.
    '''
    def __init__(self, lr = 0, degree = 1, l = 0) -> None:
        ''' 
        Initialises the model. 

        Arguments:
        lr = learning rate (kept 0 by default).
        degree = the degree of curve that needs to be fit (kept 1 by default).
        l = value of regularisation parameter (kept 0 by default).
        '''
        self._lambda = l
        self._lr = lr
        self._degree = degree
        pass

    def SetLambda(self, l) -> None:
        ''' 
        This function is for setting the regularisation parameter - lambda.\n
        Note: By default, lambda is taken to be 0.

        Arguments:
        l = the value that lambda will take.
        '''
        self._lambda = l
        return None
    
    def GetLambda(self):
        ''' 
        Returns the value of the regularisation parameter - lambda. \n
        Note: 0 is returned if no value has been set earlier.
        '''
        return self._lambda
    
    def SetLearningRate(self, lr: float) -> None:
        ''' 
        Helps in setting the learning rate for the linear regressor to use for fitting to the 
        data. (kept 0 by default)

        Arguments:
        lr = the floating value denoting the learning rate.
        '''
        self._lr = lr 
        return None
    
    def GetLearningRate(self) -> float:
        ''' 
        Returns the learning rate that the model is using.
        '''
        return self._lr
    
    def SetDegree(self, degree: int) -> None:
        ''' 
        Helps in setting the degree of the curve that needs to be fit onto data.\n
        Note: Degree is assumed to be 1 by default.

        Arguments:
        degree = value denoting degree of the fitting curve.
        '''
        self._degree = degree
        return None
    
    def GetDegree(self) -> float:
        ''' 
        Returns the degree of the fitting curve.
        '''
        return self._degree

    def DataSplitter(self, train: float, splitValid: bool, test: float, X: float, Y: float) -> float:
        ''' 
        Splits the data into train, validation and test sets based on the user defined ratios.\n
        The split is done based on stratified sampling to sustain the diversity of data in all the 
        three sets. \n
        Note: If splitValid is set to True, enter the values of train and test ratios accordingly. The 
        function will not divide the testing set into two parts if train + test ratios = 1

        Arguments:
        train = the ratio of data that needs to be set aside as training set.
        splitValid = boolean value to denote whether the validation set is required or no.
        test = the ratio of data that needs to be set aside as testing set.
        X = array of independent variables
        Y = array of dependent variables corresponding to X
        '''
        if (not splitValid and train+test != 1):
            print("The sum of train and test ratios does not equate to unity. Please enter valid ratios "
                  "or allow creation of validation set.")
            return None, None
        elif (splitValid and train+test == 1):
            print("The sum of train and test ratios equate to unity and thus, the valid set cannot be "
                  "created. Please enter appropriate ratios.")
            return None, None, None
        
        low, high = 0, len(X)
        indices = np.random.permutation(high)
        train_range = round(train*high) 
        test_range = round(test*high)
        x = X[indices]
        y = Y[indices]

        X_train = x[0:train_range]
        y_train = y[0:train_range]

        X_test = x[train_range:train_range+test_range]
        y_test = y[train_range:train_range+test_range]

        if (splitValid):
           X_valid = x[train_range+test_range:high]
           y_valid = y[train_range+test_range:high]
           return X_train, y_train, X_valid, y_valid, X_test, y_test

        return X_train, y_train, X_test, y_test 

    def Transform2Poly(self, X: float) -> float:
        ''' 
        For nth degree regression, the function converts the given set of values of the independent 
        variable to a set of values that contains all the values up to nth degree.

        Arguments:
        X = initial set of data points of the independent variable.
        ''' 
        if (self._degree == 1):
            return X
        temp = X.copy()
        for i in range(2, self._degree+1):
            X = np.c_[X, temp**i]

        return X
      
    def fit(self, X_train: float, y_train: float, method: str, epochs: int, regularise = False, reg_method = None) -> float:
        ''' 
        Train the model using the dependent (y) and independent (x) variables and any of the two 
        possible methods - closed form or gradient descent. The function also incorporates the effect
        of regularisation on the weights.\n
        Note: The code will automatically add a column of ones to the independent variable to account 
        for the bias term. 

        Arguments:
        X_train = independent variable (or values in x)
        y_train = dependent variable (or values in y)
        method = string value denoting which of the two forms of training from those mentioned in 
                 description should be used for training.
        epochs = the number of times the code needs to be run
        regularise = boolean value to denote whether regularisation is required or no.
        reg_method = the method of regularisation to be used.
        '''
        if (X_train.ndim >= 2):
            beta = np.zeros((X_train.shape[1]+1, 1), np.float32)
        elif (X_train.ndim == 1):
            beta = np.zeros((2, 1), dtype = np.float32)
        X_train = np.c_[np.ones(len(X_train)), X_train]
        y_train = y_train.reshape((y_train.shape[0], 1))
        print(beta.shape)
        if (method == 'cf'):
            if (regularise):
                if (reg_method == 'l2'):
                    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train) + 
                                                         self._lambda*np.identity(X_train.shape[1])), 
                                                         X_train.T), 
                                                         y_train)
                elif (reg_method == 'l1'):
                    print("Closed form solution does not exist for L1 regularisation due to "
                          "non-differentiability of the reguliser.")
                    return None
            else:
                beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), 
                                           X_train.T), 
                                           y_train)
            self._beta = beta
            return beta, None
        elif (method == 'gd'):
            if (self._lr == 0):
                print("Please enter the learning rate for gradient descent.")
                return None
            mse = []
            for i in range(epochs):
                y_pred = np.matmul(X_train, beta)
                if (regularise):
                    if (reg_method == 'l2'):   
                        mse.append(np.mean(np.matmul(y_pred.T - y_train.T, y_pred - y_train)) + 
                                       self._lambda*np.matmul(beta.T, beta))
                        grad = 2*np.mean(np.matmul(X_train.T, y_pred - y_train), axis = 1)
                        grad = grad.reshape((grad.shape[0], 1))
                        grad += 2*self._lambda*beta
                    elif (reg_method == 'l1'):
                        mse.append(np.mean(np.matmul(y_pred.T - y_train.T, y_pred - y_train)) + 
                                       self._lambda*np.sum(np.abs(beta)))
                        grad = 2*np.mean(np.matmul(X_train.T, y_pred - y_train), axis = 1)
                        grad = grad.reshape((grad.shape[0], 1))
                        grad += self._lambda*np.sign(beta)
                else:
                    mse.append(np.mean(np.matmul(y_pred.T - y_train.T, y_pred - y_train)))
                    grad = 2*np.mean(np.matmul(X_train.T, y_pred - y_train), axis = 1)
                    grad = grad.reshape((grad.shape[0], 1))
                beta -= self._lr*grad

            self._beta = beta
        return beta, mse[len(mse)-1], y_pred
    
    # def eval(self, X_valid: float, y_valid: float) -> float:
    #     ''' 
    #     Evaluate the model performance on the validation set.

    #     Arguments:
    #     X_valid = validation set used for evaluation
    #     y_valid = the validation set of dependent variable corresponding to the values in X_valid
    #     '''
    #     X_valid = np.c_[np.ones(len(X_valid)), X_valid]
    #     y_valid = y_valid.reshape((y_valid.shape[0], 1))
    #     y_pred = np.matmul(X_valid, self._beta)
    #     mse = np.mean(np.matmul(y_pred.T - y_valid.T, y_pred - y_valid))

    #     return y_pred, mse
    
    def predict(self, X_test, y_test):        
        ''' 
        Evaluate the model performance on the validation set.

        Arguments:
        X_valid = validation set used for evaluation
        y_valid = the validation set of dependent variable corresponding to the values in X_valid
        '''
        X_test = np.c_[np.ones(len(X_test)), X_test]
        y_test = y_test.reshape((y_test.shape[0], 1))
        y_pred = np.matmul(X_test, self._beta)
        mse = np.mean(np.matmul(y_pred.T - y_test.T, y_pred - y_test))

        return y_pred, mse