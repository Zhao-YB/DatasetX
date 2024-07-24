import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 300 
import pandas as pd
import numpy as np
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.pylab import rcParams
from matplotlib.patches import FancyArrowPatch
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV

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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

processed_x_train = pd.read_csv(r'D:\battery-forecasting-main\test\x_eis_fixed_pure_re.csv', header=None).iloc[:1128,:]
y_train = pd.read_csv(r'D:\Code\battery-forecasting-main\test\y_eis_fixed.csv', header=None).iloc[:1128,:]
processed_x_test = pd.read_csv(r'D:\Code\battery-forecasting-main\test\x_eis_fixed_pure_re.csv', header=None).iloc[1128:,:]
y_test = pd.read_csv(r'D:\Code\battery-forecasting-main\test\y_eis_fixed.csv', header=None).iloc[1128:,:]
processed_x_train = processed_x_train.iloc[:,7] - processed_x_train.iloc[:, 33]
processed_x_test = processed_x_test.iloc[:,7] - processed_x_test.iloc[:, 33]

model = XGBRegressor(learning_rate=0.1, n_estimators=50, max_depth=6, subsample=0.6, reg_alpha=1, reg_lambda=2, objective='reg:squarederror')

kf = KFold(n_splits=10, shuffle=True, random_state=0)

mae_scores = []
mape_scores = []
rmse_scores = []
r2_scores = []

for train_index, val_index in kf.split(processed_x_train):
    X_train_fold, X_val_fold = processed_x_train[train_index], processed_x_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index,:], y_train.iloc[val_index,:]

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
print(f"Average MAPE: {np.mean(mape_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"Average R2: {np.mean(r2_scores):.4f}")

y_pred_test = model.predict(processed_x_test)
y_train_pred = model.predict(processed_x_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)

print("--- test ---")
print(f"MAE: {mae_test:.4f}")
print(f"MAPE: {mape_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"R2: {r2_test:.4f}")

import sys
sys.path.append('../')
import time
import pickle

import numpy as np
from sklearn.metrics import r2_score

from utils.exp_util_pure import extract_data, extract_input
from utils.models import XGBModel

import pdb

np.random.seed(42)

file_path1 = r'D:\battery-forecasting-main\test\x_eis_variable_pure_re.csv'


df1 = pd.read_csv(file_path1, header=None)



re_feature = df1.iloc[:,22] - df1.iloc[:,26]
re_feature = np.array(re_feature).reshape(-1, 1)

feature = re_feature

channels = [1, 2, 3, 4, 5, 6, 7, 8]
params = {'max_depth':100,
          'n_splits':12,
          'n_estimators':500,
          'n_ensembles':10}
experiment = 'variable-discharge'
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
y_pred_te1_list = []
y_test1_list = []

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
        y_pred_te1_list.extend(y_pred_te1)
        y_test1_list.extend(y_test1)
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
