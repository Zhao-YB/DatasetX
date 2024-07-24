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

import numpy as np
import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pystan

def hbm_vary_intercepts_slopes_model(train_df, test_df, selected_names, individual_names, feat_dim):

    group_num = len(train_df.cluster_c4.unique())
    with_group_names = selected_names.copy()
    with_group_names.append('cluster_c4')

    x_train, x_test = train_df.loc[:, with_group_names], test_df.loc[:, with_group_names]
    y_train, y_test = train_df.loc[:, 'Lifetime'], test_df.loc[:, 'Lifetime']


    group_level_variables_train = np.zeros((group_num, 1))
    for group_idx in range(group_num):
        mean_gf = np.mean(x_train.loc[x_train.cluster_c4 == group_idx, 'avg_stress'].values)
        group_level_variables_train[group_idx, 0] = mean_gf

    raw_group_ind_train = x_train.loc[:, 'cluster_c4'].values.astype(int)
    raw_group_ind_test = x_test.loc[:, 'cluster_c4'].values.astype(int)

    transformer = StandardScaler()
    processed_x_train = transformer.fit_transform(x_train.loc[:, individual_names])
    processed_x_test = transformer.transform(x_test.loc[:, individual_names])

    stan_code = """
    data {
        int<lower=0> N; 
        int<lower=0> K;  
        int<lower=0> J;  
        int<lower=1, upper=J> group[N];  
        matrix[N, K] X; 
        vector[N] y;  
        vector[J] group_avg_stress;  
    }
    parameters {
        real g_intercept;
        real<lower=0> sigma_intercept;
        real<lower=0> sigma_slope0;
        real<lower=0> sigma_slope1;
        real<lower=0> sigma;
        vector[J] intercept_raw;
        vector[J] slope0_raw;
        vector[J] slope1_raw;
        real slope0_mu;
        real slope1_mu;
    }
    transformed parameters {
        vector[J] intercept;
        vector[J] slope0;
        vector[J] slope1;
        intercept = g_intercept + sigma_intercept * intercept_raw;
        slope0 = slope0_mu + sigma_slope0 * slope0_raw;
        slope1 = slope1_mu + sigma_slope1 * slope1_raw;
    }
    model {
        intercept_raw ~ normal(0, 1);
        slope0_raw ~ normal(0, 1);
        slope1_raw ~ normal(0, 1);
        y ~ normal(intercept[group] + slope0[group] .* X[, 1] + slope1[group] .* X[, 2], sigma);
    }
    generated quantities {
        vector[N] y_pred;
        for (n in 1:N)
            y_pred[n] = normal_rng(intercept[group[n]] + slope0[group[n]] * X[n, 1] + slope1[group[n]] * X[n, 2], sigma);
    }
    """

    stan_model = pystan.StanModel(model_code=stan_code)

    # 准备数据
    stan_data = {
        'N': processed_x_train.shape[0],
        'K': processed_x_train.shape[1],
        'J': group_num,
        'group': raw_group_ind_train + 1,  
        'X': processed_x_train,
        'y': y_train.values,
        'group_avg_stress': group_level_variables_train.flatten()
    }

    fit = stan_model.sampling(data=stan_data, iter=2000, chains=4, warmup=1000, thin=1)

    y_pred = fit.extract()['y_pred'].mean(axis=0)

    tol_mape_hierarchical = mean_absolute_percentage_error(y_true=y_test.values, y_pred=y_pred)
    tol_rmse_hierarchical = mean_squared_error(y_true=y_test.values, y_pred=y_pred, squared=False)

    return [tol_rmse_hierarchical, tol_mape_hierarchical, fit]

DATA_DIR = '../hbm_related_codes/train_test_split_revised/'
train_df = pd.read_csv(DATA_DIR + 'training.csv',index_col=0)
testin_df = pd.read_csv(DATA_DIR + 'test_in.csv',index_col=0)
testout_df = pd.read_csv(DATA_DIR + 'test_out.csv',index_col=0)

# Calculate the corresponding features mentioned in the paper
# avg_stress: Stress_avg - the group level feature
# log_mean_dqdv_dchg_mid_3_0, log_delta_CV_time_03 are the top two individual-level features mention in Table 1

                                  
train_df['log_mean_dqdv_dchg_mid_3_0'] = np.log(abs(train_df.mean_dqdv_dchg_mid_3_0))
testin_df['log_mean_dqdv_dchg_mid_3_0'] = np.log(abs(testin_df.mean_dqdv_dchg_mid_3_0))
testout_df['log_mean_dqdv_dchg_mid_3_0'] = np.log(abs(testout_df.mean_dqdv_dchg_mid_3_0))

train_df['log_delta_CV_time_03'] = np.log(abs(train_df.delta_CV_time_3_0))
testin_df['log_delta_CV_time_03'] = np.log(abs(testin_df.delta_CV_time_3_0))
testout_df['log_delta_CV_time_03'] = np.log(abs(testout_df.delta_CV_time_3_0))

train_df['log(Lifetime)'] = np.log(train_df.Lifetime)
testin_df['log(Lifetime)'] = np.log(testin_df.Lifetime)
testout_df['log(Lifetime)'] = np.log(testout_df.Lifetime)

selected_names = ['avg_stress', 'log_mean_dqdv_dchg_mid_3_0','log_delta_CV_time_03','DoD']
individual_names = [ 'log_mean_dqdv_dchg_mid_3_0','log_delta_CV_time_03','DoD']
group_num = len(train_df.cluster_c4.unique())
with_group_names = selected_names.copy()
with_group_names.append('cluster_c4')

x_train = train_df.loc[:, with_group_names]
x_test = testin_df.loc[:, with_group_names]
x_testout = testout_df.loc[:, with_group_names]

y_train = train_df.loc[:, 'Lifetime']
y_test = testin_df.loc[:, 'Lifetime']
y_testout = testout_df.loc[:, 'Lifetime']


transformer = StandardScaler()
processed_x_train = transformer.fit_transform(x_train.loc[:, individual_names])
processed_x_test = transformer.transform(x_test.loc[:, individual_names])
processed_x_testout = transformer.transform(x_testout.loc[:, individual_names])

import json
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

### Define a function to load RPT data and convert the data type ###
def convert_RPT_to_dict(RPT_json_dir,cell,subfolder):
    with open(RPT_json_dir+subfolder+cell+'.json','r') as file:
        data_dict = json.loads(json.load(file))
    
    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    for iii in range(len(data_dict['start_stop_time']['start'])):
        data_dict['QV_charge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_2']['t'][iii]))
        data_dict['QV_discharge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_2']['t'][iii]))
        data_dict['QV_charge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_5']['t'][iii]))
        data_dict['QV_discharge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_5']['t'][iii]))
    
    # Return the preprocessed Python Dictionary
    return data_dict

### Define a function to differentiate interpolated QV curves
def diff_QV(QV_int_dir,cell,subfolder,V_interpolate):
    # Transpose the preprocessed interpolated QV curves to have each row 
    # representing a week, and save as np array for easier indexing
    QV_discharge_interpolate = pd.read_csv(QV_int_dir+subfolder+cell+'.csv',header=None).T
    QV_array = QV_discharge_interpolate.to_numpy()
    
    # Create empty lists and find numerical derivatives of the QV curve
    dQdV = []; 
    for i in range(len(QV_array)):
        Q = QV_array[i]
        dQdV.append(np.diff(Q)/np.diff(V_interpolate))
    
    # Save and return as arrays for easier indexing afterward
    dQdV_array = np.array(dQdV)
    return dQdV_array

### Define a function to calculate the lifetime ###
def lifetime(capacity_fade_dir,cell,subfolder):
    # Read capacity fade from preprocessed CSV file
    capacity_fade_data = pd.read_csv(capacity_fade_dir+subfolder+cell+'.csv')
    # Raw measurements (Q and time)
    xi = capacity_fade_data['Time'].values
    yi = capacity_fade_data['Capacity'].values
    # Interpolation range of time
    x = np.arange(0,np.ceil(np.max(xi)),0.001)
    # Interpolated capacity fade
    y = pchip_interpolate(xi, yi, x)
    
    # Find the lifetime (i.e., time stamp at which y <=0.2)
    life_idx = np.argmin(np.abs(y-0.26))
    life = x[life_idx]
    return np.round(life,3)
    

if __name__ == '__main__':
    # List of cells used in this paper. Some cells from the dataset didn't 
    # reach the EOL at the time when we finalized results for this paper 
    valid_cells = pd.read_csv('valid_cells_paper.csv').values.flatten().tolist()
    
    # Cell in Release 2.0 (to determine the proper subfolder in released dataset)
    batch2 = ['G57C1','G57C2','G57C3','G57C4','G58C1', 'G26C3','G49C1','G49C2','G49C3','G49C4','G50C1','G50C3','G50C4'] 

    ### Main directory for the dataset ### TO MODIFY
    main_dir = r'G:\Predicting battery lifetime'
    # Directory of raw JSON files for RPT
    RPT_json_dir = main_dir+'/RPT_json/'
    # Directory of preprocessed (interpolated) QV curves
    QV_int_dir = main_dir+'/Q_interpolated/'
    # Directory of preprocessed capacity vs. time data
    capacity_fade_dir = main_dir+'/capacity_fade/' 
    
    
    # Create dictionaries for dQdV curves
    dQdV_dict = {}
    life = {}
  
    # 1000 equally spaced voltage points for deriving derivatives. This is for 
    # consistency with the interpolation during the preprocessing. When we extract
    # features, we use the full voltage range (3.0V to 4.2V).
    V_interpolate = np.linspace(3,4.18,1000)  
    
    # DataFrame to store the results
    results = pd.DataFrame(columns=np.arange(1000))
    
    for i,cell in enumerate(valid_cells):
        if cell in batch2:
            subfolder = 'Release 2.0/'
        else:
            subfolder = 'Release 1.0/'
        # Calculate the lifetime
        life[cell] = lifetime(capacity_fade_dir,cell,subfolder)
        # Derive differential curves
        dQdV_dict[cell] = diff_QV(QV_int_dir,cell,subfolder,V_interpolate)
        # Calculate the difference
        diff = dQdV_dict[cell][4] - dQdV_dict[cell][0]
        
        # Append the difference and lifetime to the DataFrame
        results.loc[cell] =  np.append(diff, life[cell])
    # save DataFrame to csv file
    #results.to_csv('dQdV_diff_and_life.csv')

result_train_df = results.drop(results.columns[[-1]], axis=1).iloc[:, ::10][results.index.isin(train_df['Cell'])]
result_testin_df = results.drop(results.columns[[-1]], axis=1).iloc[:, ::10][results.index.isin(testin_df['Cell'])]
result_testout_df = results.drop(results.columns[[-1]], axis=1).iloc[:, ::10][results.index.isin(testout_df['Cell'])]
y_train = results[results.columns[[-1]]][results.index.isin(train_df['Cell'])]
y_test = results[results.columns[[-1]]][results.index.isin(testin_df['Cell'])]
y_testout = results[results.columns[[-1]]][results.index.isin(testout_df['Cell'])]

x_data = result_train_df
y_data = y_train

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

