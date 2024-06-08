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

import numpy as np
import matplotlib.pyplot as plt
import pickle
batch1 = pickle.load(open(r'.\Data\batch1.pkl', 'rb'))
#remove batteries that do not reach 80% capacity
del batch1['b1c8']
del batch1['b1c10']
del batch1['b1c12']
del batch1['b1c13']
del batch1['b1c22']
numBat1 = len(batch1.keys())
batch2 = pickle.load(open(r'.\Data\batch2.pkl','rb'))
batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
add_len = [662, 981, 1060, 208, 482];
for i, bk in enumerate(batch1_keys):
    batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
    for j in batch1[bk]['summary'].keys():
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
    last_cycle = len(batch1[bk]['cycles'].keys())
    for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
        batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]
del batch2['b2c7']
del batch2['b2c8']
del batch2['b2c9']
del batch2['b2c15']
del batch2['b2c16']
numBat2 = len(batch2.keys())
batch3 = pickle.load(open(r'.\Data\batch3.pkl','rb'))
# remove noisy channels from batch3
del batch3['b3c37']
del batch3['b3c2']
del batch3['b3c23']
del batch3['b3c32']
del batch3['b3c42']
del batch3['b3c43']
numBat3 = len(batch3.keys())
numBat = numBat1 + numBat2 + numBat3
bat_dict = {**batch1, **batch2, **batch3}

import numpy as np
from scipy.interpolate import interp1d, splrep
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300 
Q100_10 = []
xx = np.arange(2.001, 3.46, 0.01)
for i in bat_dict.keys():
    if i == 'b2c1':
        Qd1 = bat_dict[i]['cycles']['12']['Qd']
        V1 = bat_dict[i]['cycles']['12']['V']

        Qd2 = bat_dict[i]['cycles']['101']['Qd']
        V2 = bat_dict[i]['cycles']['101']['V']

        # Process dataset 1
        m1 = []
        n1 = []

        for k in range(1, len(Qd1)):
            if Qd1[k] >= 1e-8 and Qd1[k] > Qd1[k-1] and V1[k] < V1[k-1] and 2.0 <= V1[k] <= 3.59:
                m1.append(Qd1[k])
                n1.append(V1[k])

        m1 = np.array(m1)
        n1 = np.array(n1)
        if not any(2 <= v <= 3.5 for v in n1):
            continue 

        try:
            interp_func1 = interp1d(n1, m1, kind='slinear')
            yy1 = interp_func1(xx)
        except ValueError:
            print('Value Range Error', i)
            continue


        yy1 = interp_func1(xx)

        # Process dataset 2
        m2 = []
        n2 = []

        for k in range(1, len(Qd2)):
            if Qd2[k] >= 1e-8 and Qd2[k] > Qd2[k-1] and V2[k] < V2[k-1] and 2.0 <= V2[k] <= 3.59:
                m2.append(Qd2[k])
                n2.append(V2[k])

        m2 = np.array(m2)
        n2 = np.array(n2)
        if not any(2.2 <= v <= 3.3 for v in n2):
            continue 

        try:
            interp_func2 = interp1d(n2, m2, kind='slinear')
            yy2 = interp_func1(xx)
        except ValueError:
            print('Value Range Error', i)
            continue

        yy2 = interp_func2(xx)
        if np.min(yy2 - yy1) < -0.13 or np.max(yy2 - yy1) > 0.01:
            print('Outlier Error', i)
            continue
        if any(np.diff(yy2-yy1) > 0.03):
            print('Jitter Error', i)
            continue  
        plt.plot(yy2-yy1,xx,linewidth=0.5)
        plt.xlim([-0.16, 0.02])
        plt.ylim([2.2, 3.3])
        Q100_10.append(yy2-yy1)
    elif i ==  'b2c17':
        Qd1 = bat_dict[i]['cycles']['10']['Qd']
        V1 = bat_dict[i]['cycles']['10']['V']

        Qd2 = bat_dict[i]['cycles']['101']['Qd']
        V2 = bat_dict[i]['cycles']['101']['V']

        # Process dataset 1
        m1 = []
        n1 = []

        for k in range(1, len(Qd1)):
            if Qd1[k] >= 1e-8 and Qd1[k] > Qd1[k-1] and V1[k] < V1[k-1] and 2.0 <= V1[k] <= 3.59:
                m1.append(Qd1[k])
                n1.append(V1[k])

        m1 = np.array(m1)
        n1 = np.array(n1)
        if not any(2 <= v <= 3.6 for v in n1):
            continue 

        try:
            interp_func1 = interp1d(n1, m1, kind='slinear')
            yy1 = interp_func1(xx)
        except ValueError:
            print('Value Range Error', i)
            continue


        yy1 = interp_func1(xx)

        # Process dataset 2
        m2 = []
        n2 = []

        for k in range(1, len(Qd2)):
            if Qd2[k] >= 1e-8 and Qd2[k] > Qd2[k-1] and V2[k] < V2[k-1] and 2.0 <= V2[k] <= 3.59:
                m2.append(Qd2[k])
                n2.append(V2[k])

        m2 = np.array(m2)
        n2 = np.array(n2)
        if not any(2.2 <= v <= 3.6 for v in n2):
            continue 

        try:
            interp_func2 = interp1d(n2, m2, kind='slinear')
            yy2 = interp_func1(xx)
        except ValueError:
            print('Value Range Error', i)
            continue

        yy2 = interp_func2(xx)
        if np.min(yy2 - yy1) < -0.13 or np.max(yy2 - yy1) > 0.01:
            print('Outlier Error', i)
            continue
        if any(np.diff(yy2-yy1) > 0.03):
            print('Jitter Error', i)
            continue  
        plt.plot(yy2-yy1,xx,linewidth=0.5)
        plt.xlim([-0.16, 0.02])
        plt.ylim([2.2, 3.3])
        Q100_10.append(yy2-yy1)
    else:
        Qd1 = bat_dict[i]['cycles']['10']['Qd']
        V1 = bat_dict[i]['cycles']['10']['V']

        Qd2 = bat_dict[i]['cycles']['100']['Qd']
        V2 = bat_dict[i]['cycles']['100']['V']

        # Process dataset 1
        m1 = []
        n1 = []

        for k in range(1, len(Qd1)):
            if Qd1[k] >= 1e-8 and Qd1[k] > Qd1[k-1] and V1[k] < V1[k-1] and 2.0 <= V1[k] <= 3.59:
                m1.append(Qd1[k])
                n1.append(V1[k])

        m1 = np.array(m1)
        n1 = np.array(n1)
        if not any(2 <= v <= 3.6 for v in n1):
            continue 

        try:
            interp_func1 = interp1d(n1, m1, kind='slinear')
            yy1 = interp_func1(xx)
        except ValueError:
            print('Value Range Error', i)
            continue


        yy1 = interp_func1(xx)

        # Process dataset 2
        m2 = []
        n2 = []

        for k in range(1, len(Qd2)):
            if Qd2[k] >= 1e-8 and Qd2[k] > Qd2[k-1] and V2[k] < V2[k-1] and 2.0 <= V2[k] <= 3.59:
                m2.append(Qd2[k])
                n2.append(V2[k])

        m2 = np.array(m2)
        n2 = np.array(n2)
        if not any(2.2 <= v <= 3.6 for v in n2):
            continue 

        try:
            interp_func2 = interp1d(n2, m2, kind='slinear')
            yy2 = interp_func1(xx)
        except ValueError:
            print('Value Range Error', i)
            continue

        yy2 = interp_func2(xx)
        
        if any(np.diff(yy2-yy1) > 0.03):
            print('Jitter Error', i)
            continue  
        plt.plot(yy2-yy1,xx,linewidth=0.5)
        plt.xlim([-0.16, 0.02])
        plt.ylim([2.15, 3.3])
        Q100_10.append(yy2-yy1)
plt.xlabel('Q100-Q10')
plt.ylabel('V')

Q100_10 = np.array(Q100_10)
cycle_life = []
for i in bat_dict.keys():
    if i == 'b2c1':
        continue
    else:
        cycle_life.append(bat_dict[i]['cycle_life'][0][0])
cycle_life = np.array(cycle_life)
var = np.var(Q100_10, axis=1)

x_data = pd.DataFrame(Q100_10, columns=[f'col_{i}' for i in range(x.shape[1])])
y_data = pd.DataFrame(cycle_life, columns=['cycle_life'])

correlation_matrix = np.zeros((x_data.shape[1], x_data.shape[1]))


for i in range(x_data.shape[1]):
    baseline_column = x_data.iloc[:, i] 
    
    for j in range(x_data.shape[1]):
        modified_column = x_data.iloc[:, j] - baseline_column
        corr, _ = pearsonr(modified_column, y_data['cycle_life'])
        correlation_matrix[j, i] = corr

result_df = pd.DataFrame(correlation_matrix, columns=[f'Column {i+1}' for i in range(x_data.shape[1])], index=[f'Column {i+1}' for i in range(x_data.shape[1])])

result_df_abs = result_df.abs()
plt.rcParams['figure.dpi'] = 300 
# Display every 10th tick label
step = 10
result_df_abs.columns = ['F$_{}$$_{}$'.format(i//10, i%10) for i in range(1, 101)]
result_df_abs.index = ['F$_{}$$_{}$'.format(i//10, i%10) for i in range(1, 101)]

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(result_df_abs, cmap='coolwarm', fmt='.2f')

# Get x and y tick locations
xticks = heatmap.get_xticks()
yticks = heatmap.get_yticks()

# Select tick labels based on the step
xtick_labels = result_df_abs.columns[::step]
ytick_labels = result_df_abs.index[::step]

# Set x and y tick locations and labels
heatmap.set_xticks(xticks[::5])
heatmap.set_xticklabels(xtick_labels, fontsize=15)

heatmap.set_yticks(yticks[::5])
heatmap.set_yticklabels(ytick_labels, fontsize=15)

result_df_abs = result_df_abs.fillna(0)
max_value_index  = result_df_abs.values.argmax()
print(divmod(max_value_index, result_df_abs.shape[1]))
print(result_df_abs.values.max())