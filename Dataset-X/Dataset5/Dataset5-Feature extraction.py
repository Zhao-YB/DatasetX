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

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
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

import pandas as pd
import os
import re
import numpy as np

df_res1 = pd.DataFrame(columns=['cycle', 'Voltages', 'rate', 'Tem', 'Capacity'])
files = os.listdir('./Dataset_1_NCA_battery/')
for file in range(len(files)):
    Tem = int(files[file][2:4])
    data_r = pd.read_csv(os.path.join('./Dataset_1_NCA_battery/', files[file]))
    for i in range(int(np.min(data_r['cycle number'].values)), int(np.max(data_r['cycle number'].values))+1):
        data_i = data_r[data_r['cycle number'] == i]
        Ecell = np.array(data_i['Ecell/V'])
        Q_dis = np.array(data_i['Q discharge/mA.h'])
        Current = np.array(data_i['<I>/mA'])
        control = np.array(data_i['control/V/mA'])
        cr = np.array(data_i['control/mA'])[1]/3500
        if np.max(Q_dis) < 2500 or np.max(Q_dis) > 3500:
            continue
        index = np.where(np.abs(control) == 0)
        start = index[0][0]
        end = 13
        for j in range(3):
            if control[start+3] == 0:
                break
            else:
                start = index[0][j+1]
        if Current[start] > 1:
            start = start + 1
            if control[start + 13] != 0:
                end = 12
        if control[start + end] == 0 and Ecell[start + end] > 4.0:
            df_res1 = df_res1.append({'cycle': i, 'Voltages': Ecell[start:start+14], 'rate': cr, 'Tem': Tem,
                                    'Capacity': np.max(Q_dis)}, ignore_index=True)

print('Features extraction is done')

df_res2 = pd.DataFrame(columns=['cycle', 'Voltages', 'rate', 'Tem', 'Capacity'])
files = os.listdir('./Dataset_2_NCM_battery/')
for file in range(len(files)):
    Tem = int(files[file][2:4])
    data_r = pd.read_csv(os.path.join('./Dataset_2_NCM_battery/', files[file]))
    for i in range(int(np.min(data_r['cycle number'].values)), int(np.max(data_r['cycle number'].values))+1):
        data_i = data_r[data_r['cycle number'] == i]
        Ecell = np.array(data_i['Ecell/V'])
        Q_dis = np.array(data_i['Q discharge/mA.h'])
        Current = np.array(data_i['<I>/mA'])
        control = np.array(data_i['control/V/mA'])
        cr = np.array(data_i['control/mA'])[1] / 3500
        if np.max(Q_dis) < 2500:
            continue
        index = np.where(np.abs(control) == 0)
        if index[0][0] > 0:
            start = index[0][0]
        else:
            start = index[0][14]
            print(i)
        if control[start + 13] == 0:
            df_res2 = df_res2.append(
                {'cycle': i, 'Voltages': Ecell[start:start + 14], 'rate': cr, 'Tem': Tem, 'Capacity': np.max(Q_dis)},
                ignore_index=True)

# Save to excel file

# Or save to csv file
# df_res.to_csv('Dataset_2_NCM_battery.csv', index=False)
print('Features extraction is done')

df_res3 = pd.DataFrame(columns=['cycle', 'Voltages', 'C_rate', 'D_rate', 'Tem', 'Capacity'])
files = os.listdir('./Dataset_3_NCM_NCA_battery/')
for file in range(len(files)):
    Tem = int(files[file][2:4])
    data_r = pd.read_csv(os.path.join('./Dataset_3_NCM_NCA_battery/', files[file]))
    k = np.min(data_r['cycle number'].values)
    data_k = data_r[data_r['cycle number'] == k]
    Q_p = np.max(np.array(data_k['Q discharge/mA.h']))
    delta = 1
    for i in range(int(np.min(data_r['cycle number'].values)), int(np.max(data_r['cycle number'].values))+1):
        data_i = data_r[data_r['cycle number'] == i]
        Ecell = np.array(data_i['Ecell/V'])
        Q_dis = np.array(data_i['Q discharge/mA.h'])
        Current = np.array(data_i['<I>/mA'])
        control = np.array(data_i['control/V/mA'])
        cr = np.array(data_i['control/mA'])[1] / 2500
        cr_d = int(files[file][8])
        if np.max(Q_dis) < 1650 or np.max(Q_dis) > 2510:
            delta = delta + 1
            continue
        # Remove points where capacity changes too quickly
        if np.abs(np.max(Q_dis) - Q_p) > delta * 10:
            delta = delta + 1
            continue
        delta = 1
        Q_p = np.max(Q_dis)
        index = np.where(np.abs(control) == 0)
        if index[0][0] > 0:
            start = index[0][0]
        else:
            start = index[0][14]
            print(i)
        if control[start + 19] == 0:
            df_res3 = df_res3.append(
                {'cycle': i, 'Voltages': Ecell[start:start + 59], 'C_rate': cr, 'D_rate': cr_d, 'Tem': Tem,
                 'Capacity': np.max(Q_dis)}, ignore_index=True)

# Save to excel file

# Or save to csv file
# df_res.to_csv('Dataset_3_NCM_NCA_battery.csv', index=False)
print('Features extraction is done')

lens = df_res1.shape[0]
df = pd.DataFrame(columns=[ 'max', 'mean', 'min', 'var', 'ske', 'kur', 'capacity'])
for i in range(lens):
    df = df.append({'max':df_res1.Voltages[i].max(), 'mean':df_res1.Voltages[i].mean(), 'min':df_res1.Voltages[i].min(), 
                   'var':df_res1.Voltages[i].var(), 'ske':skew(df_res1.Voltages[i]), 'kur':kurtosis(df_res1.Voltages[i]) - 3,
                  'capacity':df_res1.Capacity[i]},ignore_index=True)
    
file_path1 = r'D:\data-driven-capacity-estimation-from-voltage-relaxation-main\Dataset_1_NCA_battery_poly.csv'
file_path2 = r'D:\Code\data-driven-capacity-estimation-from-voltage-relaxation-main\Dataset_1_NCA_battery_capacity.csv'

df_res_poly = pd.read_csv(file_path1) 
conditions = [
    (df_res1['Tem'] == 25) & (df_res1['rate'] == 0.25),
    (df_res1['Tem'] == 25) & (df_res1['rate'] == 0.5),
    (df_res1['Tem'] == 25) & (df_res1['rate'] == 1),
    (df_res1['Tem'] == 35) & (df_res1['rate'] == 0.5),
    (df_res1['Tem'] == 45) & (df_res1['rate'] == 0.5),
]

train_df = pd.DataFrame()
test_df = pd.DataFrame()


for condition in conditions:

    temp_df = df_res1.loc[condition]
    temp_df_poly = df_res_poly.loc[temp_df.index]


    temp_df.reset_index(drop=True, inplace=True)
    temp_df_poly.reset_index(drop=True, inplace=True)


    X_train, X_test, _, _ = train_test_split(
        temp_df_poly, temp_df['Capacity'], test_size=0.2, random_state=42
    )

    X_train['Capacity'] = temp_df.loc[X_train.index, 'Capacity']
    X_test['Capacity'] = temp_df.loc[X_test.index, 'Capacity']

    train_df = pd.concat([train_df, X_train], ignore_index=True)
    test_df = pd.concat([test_df, X_test], ignore_index=True)

x_data = train_df.drop(['Capacity'], axis=1)
y_data = train_df['Capacity']


x_data = x_data - x_data.iloc[0]
x_data = x_data.iloc[1:].reset_index(drop=True)


y_data = y_data.iloc[1:].reset_index(drop=True)


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
divmod(max_value_index, result_df_abs.shape[1])