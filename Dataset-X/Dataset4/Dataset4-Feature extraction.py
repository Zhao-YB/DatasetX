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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, mean_squared_error
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

cap_list = ['Cap_25C01','Cap_25C02','Cap_25C03' ,'Cap_25C04','Cap_25C05','Cap_25C06',
            'Cap_25C07', 'Cap_25C08','Cap_35C01','Cap_35C02','Cap_45C01','Cap_45C02']

dfs = []

for filename in cap_list:
    df = pd.read_excel('./dataset/' + filename + '.xlsx', header=None)
    dfs.append(df)

cap_df = pd.concat(dfs, ignore_index=True)

file5_lists = ['EIS_state_V_25C01','EIS_state_V_25C02','EIS_state_V_25C03','EIS_state_V_25C04','EIS_state_V_25C05',
             'EIS_state_V_25C06','EIS_state_V_25C07', 'EIS_state_V_25C08','EIS_state_V_35C01','EIS_state_V_35C02',
             'EIS_state_V_45C01','EIS_state_V_45C02']

nums = [200, 250, 229, 81, 275, 212, 140, 36, 299, 299, 299, 310]

base_path = '.\dataset\EIS data\EIS data'

def reshape_data(df):

    if len(df) % 60 != 0:
        raise ValueError("Number of rows is not divisible by 60.")
    
    new_rows = len(df) // 60
    
    new_columns = 120
    
    reshaped_data = [np.concatenate([df.iloc[i:i+60, 3].values, df.iloc[i:i+60, 4].values]) for i in range(0, len(df), 60)]
    
    new_df = pd.DataFrame(reshaped_data, columns=[f'Column_{i+1}' for i in range(new_columns)])
    
    return new_df


all_reshaped_data = []


for i, file_name in enumerate(file5_lists):

    file_path = os.path.join(base_path, file_name + '.txt')
    
    df = pd.read_csv(file_path, delimiter='\s+')
    
    limited_rows = nums[i] * 60
    df = df.head(limited_rows)
    
    reshaped_df = reshape_data(df)
    
    all_reshaped_data.append(reshaped_df)

V_df = pd.concat(all_reshaped_data, axis=0, ignore_index=True)

V_re = V_df.iloc[:, :60]

V_im = V_df.iloc[:, 60:]

x_data = V_re
y_data = cap_df

correlation_matrix = np.zeros((x_data.shape[1], x_data.shape[1]))


for i in range(x_data.shape[1]):
    baseline_column = x_data.iloc[:, i]  
    

    for j in range(x_data.shape[1]):
        modified_column = x_data.iloc[:, j] - baseline_column
        corr, _ = pearsonr(modified_column, y_data)
        correlation_matrix[j, i] = corr


result_df = pd.DataFrame(correlation_matrix, columns=[f'Column {i+1}' for i in range(x_data.shape[1])], index=[f'Column {i+1}' for i in range(x_data.shape[1])])
result_df_abs = result_df.abs()

result_df_abs = result_df_abs.fillna(0)
max_value_index  = result_df_abs.values.argmax()
print(divmod(max_value_index, result_df_abs.shape[1]))
print(result_df_abs.values.max())