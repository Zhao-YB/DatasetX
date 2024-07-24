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

y_train = np.log10(cycle_lives_train)
y_test1 = np.log10(cycle_lives_test1)
y_test2 = np.log10(cycle_lives_test2)
data_test1_mod = data_test1.copy()
cycle_lives_test1_mod = cycle_lives_test1.copy()
y_test1_mod = y_test1.copy()

data_test1_mod = np.delete(data_test1_mod, 21, axis=0)
cycle_lives_test1_mod = np.delete(cycle_lives_test1_mod, 21)
y_test1_mod = np.delete(y_test1_mod, 21)

Vdlin = np.linspace(3.6, 2, 1000)
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
alphas = np.logspace(0.001, 100, 20)
colors_list = ['Blues', 'Reds', 'Oranges']
np.random.seed(0)

data1 = data_train[:, ::10, 98] - data_train[:, ::10, 8]
data2 = data_test1_mod[:, ::10, 98] - data_test1_mod[:, ::10, 8]
data3 = data_test2[:, ::10, 98] - data_test2[:, ::10, 8]
x = np.vstack((data1, data2, data3))
y = np.vstack((cycle_lives_train.reshape(-1,1), cycle_lives_test1_mod.reshape(-1,1), cycle_lives_test2.reshape(-1,1)))

x_data = pd.DataFrame(x, columns=[f'col_{i}' for i in range(x.shape[1])])
y_data = pd.DataFrame(y, columns=['cycle_life'])

correlation_matrix = np.zeros((x_data.shape[1], x_data.shape[1]))


for i in range(x_data.shape[1]):
    baseline_column = x_data.iloc[:, i]  
    
    for j in range(x_data.shape[1]):
        modified_column = x_data.iloc[:, j] - baseline_column
        corr, _ = pearsonr(modified_column, y_data['cycle_life'])
        correlation_matrix[j, i] = corr

result_df = pd.DataFrame(correlation_matrix, columns=[f'Column {i+1}' for i in range(x_data.shape[1])], index=[f'Column {i+1}' for i in range(x_data.shape[1])])

result_df_abs = result_df_abs.fillna(0)
max_value_index  = result_df_abs.values.argmax()
print(divmod(max_value_index, result_df_abs.shape[1]))
print(result_df_abs.values.max())