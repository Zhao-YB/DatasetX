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
from nptdms import TdmsFile
from matplotlib.pylab import rcParams
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
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

import os, sys

if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('../')
    sys.path.insert(0, 'src/')

import numpy as np
import pandas as pd
import glob, re, pprint, random
from datetime import datetime
import pprint
import yaml

from scipy.stats import pearsonr, ttest_ind
from scipy.signal import savgol_filter
from scipy import interpolate

from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import pickle

with open("dataframes.pickle", "rb") as f:
    dataframes = pickle.load(f)

print("dataframes loaded。")

split_dataframes_dict = {}
for i in range(17):
    folder_name = f"CU{i:03d}"
    split_dataframes_dict[folder_name] = {} 
    for index_name, df in dataframes[folder_name].items():
        df = df[df['Command'] == 'Pause']
        diff = df.index.to_series().diff()
        grouped = df.groupby(diff.gt(1).cumsum())
        split_dfs = [group for _, group in grouped]
        split_dataframes_dict[folder_name][index_name] = split_dfs 



df_results = pd.DataFrame()

cu000_index_names = dataframes["CU000"].keys()

global_min_index = float('-inf')
global_max_index = float('inf')
for i in range(17):
    folder_name = f"CU{i:03d}"
    for index_name, df in split_dataframes_dict[folder_name].items():
        df = df[1].reset_index() 

        df['Time'] = (df['Time'] - df['Time'].iloc[0]) * 3600  
        
        df['index'] = df['index'] - df['index'].min() 
        
        global_min_index = max(global_min_index, df['Time'].min())
        global_max_index = min(global_max_index, df['Time'].max())

for i in range(1, 17):
    folder_name = f"CU{i:03d}"

    for index_name in cu000_index_names:
        if index_name in dataframes[folder_name]:
            df0 = split_dataframes_dict["CU000"][index_name][1].reset_index()
            df1 = split_dataframes_dict[folder_name][index_name][1].reset_index()
            
            df0['Time'] = (df0['Time'] - df0['Time'].iloc[0]) * 3600
            df1['Time'] = (df1['Time'] - df1['Time'].iloc[0]) * 3600
            
            df0['index'] = df0['index'] - df0['index'].min()
            df1['index'] = df1['index'] - df1['index'].min()

            f0 = interp1d(df0['Time'], df0['U'], kind='slinear')
            f1 = interp1d(df1['Time'], df1['U'], kind='slinear')

            interpolation_points = np.arange(0, 7199, 72)
            df0_interp_u = f0(interpolation_points)
            df1_interp_u = f1(interpolation_points)
            
            result = df1_interp_u - df0_interp_u

            temp_df = pd.DataFrame([result], columns=np.arange(len(result)))
            temp_df['Ah'] = df1['Ah'].max()

            df_results = df_results.append(temp_df, ignore_index=True)

x_data = df_results.drop('Ah', axis=1)
y_data = df_results['Ah']

correlation_matrix = np.zeros((x_data.shape[1], x_data.shape[1]))


for i in range(x_data.shape[1]):
    baseline_column = x_data.iloc[:, i]  
    
    for j in range(x_data.shape[1]):
        modified_column = x_data.iloc[:, j] - baseline_column
        corr, _ = pearsonr(modified_column, y_data)
        correlation_matrix[j, i] = corr

result_df = pd.DataFrame(correlation_matrix, columns=[f'Column {i+1}' for i in range(x_data.shape[1])], index=[f'Column {i+1}' for i in range(x_data.shape[1])])

result_df_abs = result_df_abs.fillna(0)
max_value_index  = result_df_abs.values.argmax()
print(divmod(max_value_index, result_df_abs.shape[1]))
print(result_df_abs.values.max())

X = x_data.iloc[:, 85] - x_data.iloc[:, 97]
X = X.values.reshape(-1, 1)  

y = y_data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBRegressor(learning_rate=0.6, n_estimators=6, max_depth=6, subsample=1, reg_alpha=0, reg_lambda=1, objective='reg:squarederror')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
mape_scores = []
rmse_scores = []
r2_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model.fit(X_train_fold, y_train_fold)

    y_pred = model.predict(X_val_fold)

    mae = mean_absolute_error(y_val_fold, y_pred)
    mape = mean_absolute_percentage_error(y_val_fold, y_pred)
    rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
    r2 = r2_score(y_val_fold, y_pred)

    mae_scores.append(mae)
    mape_scores.append(mape)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

print(f"Average MAE: {np.mean(mae_scores):.4f}")
print(f"Average MAPE: {np.mean(mape_scores):.4f}%")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"Average R2: {np.mean(r2_scores):.4f}")


y_pred_test = model.predict(X_test)
y_train_pred = model.predict(X_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)
