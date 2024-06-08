import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 300 
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.pylab import rcParams
from matplotlib.patches import FancyArrowPatch
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
from sklearn import cluster
from collections import Counter

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

import xgboost as xgb
from xgboost import XGBClassifier

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

import sys
sys.path.append('../')
import time
import pickle

import numpy as np
from sklearn.metrics import r2_score

from utils.exp_util import extract_data, extract_input
from utils.models import XGBModel

import pdb

file_path1 = r'D:\项目\燃料电池\Code\battery-forecasting-main\test\x_eis_fixed_im.csv'
file_path2 = r'D:\项目\燃料电池\Code\battery-forecasting-main\test\x_eis_fixed_re.csv'

df1 = pd.read_csv(file_path1, header=None)
df2 = pd.read_csv(file_path2, header=None)

im_feature = df1.iloc[:,30] - df1.iloc[:,56]
im_feature = np.array(im_feature).reshape(-1, 1)

re_feature = df2.iloc[:,43] - df2.iloc[:,84]
re_feature = np.array(re_feature).reshape(-1, 1)

feature = np.concatenate((im_feature, re_feature), axis=1)

channels = [1, 2, 3, 4, 5, 6, 7, 8]
params = {'max_depth':100,
          'n_splits':12,
          'n_estimators':500,
          'n_ensembles':10}
experiment = 'fixed-discharge'
log_name = '../results/{}/log-n-cells.txt'.format(experiment)
input_name = 'actions'
n_cells_list = [2, 4, 8, 16, 20]
p = np.random.permutation(16)
# Extract variable discharge data set
cell_var, cap_ds_var, data_var = extract_data(experiment, channels)
cell_ids = np.unique(cell_var)[p]
x = extract_input(input_name, data_var)

# output = next cycle discharge capacity
y = cap_ds_var
n_splits = params['n_splits']

x = np.concatenate((x, feature), axis=1)

for n_cells in n_cells_list:
    experiment_name = '{}_{}cells_xgb'.format(input_name, n_cells)
    experiment_info = '\nInput: {} \tOutput: Q_n+1 \t{} cells \nMax depth: {}\t N estimators: {}\t N ensembles: {}\tSplits:{}\n'.format(input_name, n_cells, params['max_depth'], params['n_estimators'],
                                                                                                                                        params['n_ensembles'], params['n_splits'])
    t0 = time.time()
    r2s_tr = []
    r2s_te = []
    pes_tr = []
    pes_te = []

    for split in range(n_splits):
        cell_ids_s = cell_ids[0:n_cells+2]
        experiment_info = '\nInput: {} \tOutput: c(discharge)_n+1 \nMax depth: {}\tSplits:{}\n'.format(input_name, params['max_depth'], n_cells)
        print(cell_ids[0])
        print(cell_ids[1])
        cell_test1 = cell_ids[0]
        cell_test2 = cell_ids[1]
        cell_train = cell_ids[2:n_cells+2]
        idx_test1 = np.where(np.isin(cell_var, cell_test1))
        idx_test2 = np.where(np.isin(cell_var, cell_test2))
        idx_train = np.where(np.isin(cell_var, cell_train))
        x_train = x[idx_train]
        print('Number of datapoints = {}'.format(x_train.shape[0]))
        y_train = cap_ds_var[idx_train]
        x_test1 = x[idx_test1]
        y_test1 = cap_ds_var[idx_test1]
        x_test2 = x[idx_test2]
        y_test2 = cap_ds_var[idx_test2]

        regressor = XGBModel(None, None, cell_ids_s, experiment, experiment_name, n_ensembles=params['n_ensembles'],
                             n_splits=params['n_splits'], max_depth=params['max_depth'],
                             n_estimators=params['n_estimators'])
        y_pred_tr, y_pred_tr_err, y_pred_te1, y_pred_te1_err, y_pred_te2, y_pred_te2_err, _, _, _, _ = regressor.train_and_predict(x_train, y_train, x_test1, cell_test1, X_test2=x_test2, cell_test2=cell_test2)
        r2s_tr.append(r2_score(y_train, y_pred_tr))
        r2s_te.append(r2_score(y_test1, y_pred_te1))
        r2s_te.append(r2_score(y_test2, y_pred_te2))
        pes_tr.append(np.abs(y_train - y_pred_tr) / y_train)
        pes_te.append(np.abs(y_test1 - y_pred_te1) / y_test1)
        pes_te.append(np.abs(y_test2 - y_pred_te2) / y_test2)
        cell_ids = np.roll(cell_ids, 2)

    r2_tr = np.median(np.array(r2s_tr))
    r2_te = np.median(np.array(r2s_te))
    pe_tr = 100*np.median(np.hstack(pes_tr).reshape(-1))
    pe_te = 100*np.median(np.hstack(pes_te).reshape(-1))
    print('Train R2:{}\t Train error: {}\t Test R2: {}\t Test error: {}'.format(r2_tr, pe_tr, r2_te, pe_te))

print('Done.')

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Calculate RMSE for training set
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_tr))
# Calculate RMSE for test set 1
RMSE_test1 = np.sqrt(mean_squared_error(y_test1, y_pred_te1))
# Calculate RMSE for test set 2
RMSE_test2 = np.sqrt(mean_squared_error(y_test2, y_pred_te2))

# Calculate average test RMSE
RMSE_test_avg = (RMSE_test1 + RMSE_test2) / 2

# Calculate MAPE for training set
MAPE_train = mean_absolute_percentage_error(y_train, y_pred_tr)
# Calculate MAPE for test set 1
MAPE_test1 = mean_absolute_percentage_error(y_test1, y_pred_te1)
# Calculate MAPE for test set 2
MAPE_test2 = mean_absolute_percentage_error(y_test2, y_pred_te2)

# Calculate average test MAPE
MAPE_test_avg = (MAPE_test1 + MAPE_test2) / 2

# Print the results
print("Training RMSE:", RMSE_train)
print("Test set 1 RMSE:", RMSE_test1)
print("Test set 2 RMSE:", RMSE_test2)
print("Average Test RMSE:", RMSE_test_avg)

print("Training MAPE:", MAPE_train)
print("Test set 1 MAPE:", MAPE_test1)
print("Test set 2 MAPE:", MAPE_test2)
print("Average Test MAPE:", MAPE_test_avg)