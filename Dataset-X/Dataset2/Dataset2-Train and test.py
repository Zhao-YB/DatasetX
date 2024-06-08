import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 300 
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pylab import rcParams
from scipy.stats import pearsonr, spearmanr, skew, kurtosis

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV, cross_val_predict

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score,mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn import cluster
from collections import Counter

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
#from mlxtend.plotting import plot_decision_regions #用于画决策边界

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.svm import SVC, LinearSVC, NuSVC

from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, ADASYN
from imblearn.datasets import make_imbalance
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier

import time
from functools import reduce

import copy
from functools import reduce
import glob
import json
import os
from pathlib import Path

def sortKeyFunc(s):
    return int(os.path.basename(s)[4:-4])

def load_dataset(folder):
    files = glob.glob(f'./Data/data/{folder}/*.csv')
    files.sort(key=sortKeyFunc) # glob returns list with arbitrary order
    
    l = len(files)
    dataset = np.zeros((l, 1000, 99))
    
    for k, file in enumerate(files):
        cell = np.genfromtxt(file, delimiter=',')
        dataset[k,:,:] = cell # flip voltage dimension
    
    return dataset

data_train = load_dataset('train')
data_test1 = load_dataset('test1')
data_test2 = load_dataset('test2')

cycle_lives_train = np.genfromtxt('./Data/data/cycle_lives/train_cycle_lives.csv', delimiter=',')
cycle_lives_test1 = np.genfromtxt('./Data/data/cycle_lives/test1_cycle_lives.csv', delimiter=',')
cycle_lives_test2 = np.genfromtxt('./Data/data/cycle_lives/test2_cycle_lives.csv', delimiter=',')

y_train = np.log10(cycle_lives_train)
y_test1 = np.log10(cycle_lives_test1)
y_test2 = np.log10(cycle_lives_test2)
data_test1_mod = data_test1.copy()
cycle_lives_test1_mod = cycle_lives_test1.copy()
y_test1_mod = y_test1.copy()

data_test1_mod = np.delete(data_test1_mod, 21, axis=0)
cycle_lives_test1_mod = np.delete(cycle_lives_test1_mod, 21)
y_test1_mod = np.delete(y_test1_mod, 21)
print(np.median(cycle_lives_train))
print(np.median(cycle_lives_test1_mod))
print(np.median(cycle_lives_test2))

def get_RMSE_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred):
    """
    Calculate RMSE for three datasets. Use units of cycles instead of log(cycles)
    """
    
    RMSE_train = mean_squared_error(np.power(10, y_train), np.power(10, y_train_pred), squared=False)
    RMSE_test1 = mean_squared_error(np.power(10, y_test1_mod), np.power(10, y_test1_pred), squared=False)
    RMSE_test2 = mean_squared_error(np.power(10, y_test2), np.power(10, y_test2_pred), squared=False)

    return RMSE_train, RMSE_test1, RMSE_test2

def get_MAPE_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred):
    """
    Calculate MAPE for three datasets. Use units of cycles instead of log(cycles)
    """
    
    MAPE_train = mean_absolute_percentage_error(np.power(10, y_train), np.power(10, y_train_pred))
    MAPE_test1 = mean_absolute_percentage_error(np.power(10, y_test1_mod), np.power(10, y_test1_pred))
    MAPE_test2 = mean_absolute_percentage_error(np.power(10, y_test2), np.power(10, y_test2_pred))

    return MAPE_train, MAPE_test1, MAPE_test2

from sklearn.metrics import r2_score

def get_R2_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred):
    """
    Calculate R^2 for three datasets. Use units of cycles instead of log(cycles)
    """
    
    R2_train = r2_score(np.power(10, y_train), np.power(10, y_train_pred))
    R2_test1 = r2_score(np.power(10, y_test1_mod), np.power(10, y_test1_pred))
    R2_test2 = r2_score(np.power(10, y_test2), np.power(10, y_test2_pred))

    return R2_train, R2_test1, R2_test2


from sklearn.metrics import mean_absolute_error

def get_Test_Error_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred):
    """
    Calculate Test Error (MAE) for three datasets. Use units of cycles instead of log(cycles)
    """
    
    Test_Error_train = mean_absolute_error(np.power(10, y_train), np.power(10, y_train_pred))
    Test_Error_test1 = mean_absolute_error(np.power(10, y_test1_mod), np.power(10, y_test1_pred))
    Test_Error_test2 = mean_absolute_error(np.power(10, y_test2), np.power(10, y_test2_pred))

    return Test_Error_train, Test_Error_test1, Test_Error_test2

Vdlin = np.linspace(3.6, 2, 1000)
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
alphas = np.logspace(0.001, 100, 20)
colors_list = ['Blues', 'Reds', 'Oranges']
np.random.seed(0)

def factors(n):
    return sorted(list(reduce(list.__add__,
                              ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))
sampling_frequencies = factors(1000)[:-1] # exclude 1000
mV_frequencies = (3.6 - 2.0) / (1000 / np.array(sampling_frequencies)) * 1000 # mV
RMSE_train = np.zeros((len(sampling_frequencies, )))
RMSE_test1 = np.zeros((len(sampling_frequencies, )))
RMSE_test2 = np.zeros((len(sampling_frequencies, )))
MAPE_train = np.zeros((len(sampling_frequencies, )))
MAPE_test1 = np.zeros((len(sampling_frequencies, )))
MAPE_test2 = np.zeros((len(sampling_frequencies, )))
R2_train = np.zeros((len(sampling_frequencies, )))
R2_test1 = np.zeros((len(sampling_frequencies, )))
R2_test2 = np.zeros((len(sampling_frequencies, )))
ERR_train = np.zeros((len(sampling_frequencies, )))
ERR_test1 = np.zeros((len(sampling_frequencies, )))
ERR_test2 = np.zeros((len(sampling_frequencies, )))

for k, freq in enumerate(sampling_frequencies):
    
    # Define the log10 of the variance of Q100 - Q10, with different sampling frequencies:
    X_train = np.log10(np.var((data_train[:, ::freq, 98] - data_train[:, ::freq, 8]), axis=1))
    X_test1 = np.log10(np.var((data_test1_mod[:, ::freq, 98] - data_test1_mod[:, ::freq, 8]), axis=1))
    X_test2 = np.log10(np.var((data_test2[:, ::freq, 98] - data_test2[:, ::freq, 8]), axis=1))

    # Scale via standarization:
    scaler = StandardScaler().fit(X_train.reshape(-1, 1))
    X_train_scaled = scaler.transform(X_train.reshape(-1, 1))
    X_test1_scaled = scaler.transform(X_test1.reshape(-1, 1))
    X_test2_scaled = scaler.transform(X_test2.reshape(-1, 1))

    # Define and fit linear regression via enet
    model = XGBRegressor(learning_rate=0.03, n_estimators=215, max_depth=5, subsample=1, reg_alpha=0, reg_lambda=1, objective='reg:squarederror')
#y_train_pred = cross_val_predict(model, X_train_scaled.reshape(-1, 1), y_train, cv=10)
    model.fit(X_train_scaled.reshape(-1, 1), y_train)

    # Predict on test sets
    y_train_pred = model.predict(X_train_scaled)
    y_test1_pred = model.predict(X_test1_scaled)
    y_test2_pred = model.predict(X_test2_scaled)
        
    # Evaluate error
    RMSE_train[k], RMSE_test1[k], RMSE_test2[k] = get_RMSE_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
    MAPE_train[k], MAPE_test1[k], MAPE_test2[k] = get_MAPE_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
    R2_train[k], R2_test1[k], R2_test2[k] = get_R2_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
    ERR_train[k], ERR_test1[k], ERR_test2[k] = get_Test_Error_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
    
    if k == 0:
        #save these results for histogram plot
        severson_variance_model_train_pred = y_train_pred
        severson_variance_model_test1_pred = y_test1_pred
        severson_variance_model_test2_pred = y_test2_pred

# Define the log10 of the variance of Q100 - Q10, with different sampling frequencies:
X_train = np.log10((data1[:, 60] - data1[:, 49]))
X_test1 = np.log10((data2[:, 60] - data2[:, 49]))
X_test2 = np.log10((data3[:, 60] - data3[:, 49]))

# Scale via standarization:
scaler = StandardScaler().fit(X_train.reshape(-1, 1))
X_train_scaled = scaler.transform(X_train.reshape(-1, 1))
X_test1_scaled = scaler.transform(X_test1.reshape(-1, 1))
X_test2_scaled = scaler.transform(X_test2.reshape(-1, 1))

# Define and fit linear regression via enet
model = XGBRegressor(learning_rate=0.03, n_estimators=215, max_depth=5, subsample=1, reg_alpha=0, reg_lambda=1, objective='reg:squarederror')
#y_train_pred = cross_val_predict(model, X_train_scaled.reshape(-1, 1), y_train, cv=10)
model.fit(X_train_scaled.reshape(-1, 1), y_train)

# Predict on test sets
y_train_pred = model.predict(X_train_scaled)
y_test1_pred = model.predict(X_test1_scaled)
y_test2_pred = model.predict(X_test2_scaled)

# Evaluate error
RMSE_train, RMSE_test1, RMSE_test2 = get_RMSE_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
MAPE_train, MAPE_test1, MAPE_test2 = get_MAPE_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
R2_train, R2_test1, R2_test2 = get_R2_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
ERR_train, ERR_test1, ERR_test2 = get_Test_Error_for_all_datasets(y_train_pred, y_test1_pred, y_test2_pred)
