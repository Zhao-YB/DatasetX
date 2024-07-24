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

feature = V_im.iloc[:,22] - V_im.iloc[:,32]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
X_train = feature[:760]
y_train = cap_df[:760]

model = XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)

mae = mean_absolute_error(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mape = np.mean(np.abs((y_train.values.reshape(1,-1)[0] - y_pred) / y_train.values)) * 100
r2 = r2_score(y_train, y_pred)

'''25C05'''

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
X_train = feature[:760]
y_train = cap_df[:760]
X_test = feature[760:1035]
y_test5 = cap_df[760:1035]


model = XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100)
model.fit(X_train, y_train)

y_pred5 = model.predict(X_test)

mae = mean_absolute_error(y_test5, y_pred5)
rmse = np.sqrt(mean_squared_error(y_test5, y_pred5))
mape = np.mean(np.abs((y_test5.values.reshape(1,-1)[0] - y_pred5) / y_test5.values)) * 100
r2 = r2_score(y_test5, y_pred5)

'''25C06'''
X_train = feature[:760]
y_train = cap_df[:760]
X_test = feature[1035:1247]
y_test6 = cap_df[1035:1247]

model = XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100)
model.fit(X_train, y_train)


y_pred6 = model.predict(X_test)
y_train_pred = model.predict(X_train)

mae = mean_absolute_error(y_test6, y_pred6)
rmse = np.sqrt(mean_squared_error(y_test6, y_pred6))
mape = np.mean(np.abs((y_test6.values.reshape(1,-1)[0] - y_pred6) / y_test6.values)) * 100
r2 = r2_score(y_test6, y_pred6)

'''25C07'''
X_train = feature[:760]
y_train = cap_df[:760]
X_test = feature[1247:1387]
y_test7 = cap_df[1247:1387]

model = XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100)
model.fit(X_train, y_train)

y_pred7 = model.predict(X_test)

mae = mean_absolute_error(y_test7, y_pred7)
rmse = np.sqrt(mean_squared_error(y_test7, y_pred7))
mape = np.mean(np.abs((y_test7.values.reshape(1,-1)[0] - y_pred7) / y_test7.values)) * 100
r2 = r2_score(y_test7, y_pred7)

'''25C08'''
X_train = feature[:760]
y_train = cap_df[:760]
X_test = feature[1387:1423]
y_test8 = cap_df[1387:1423]

model = XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100)
model.fit(X_train, y_train)

y_pred8 = model.predict(X_test)

mae = mean_absolute_error(y_test8, y_pred8)
rmse = np.sqrt(mean_squared_error(y_test8, y_pred8))
mape = np.mean(np.abs((y_test8.values.reshape(1,-1)[0] - y_pred8) / y_test8.values)) * 100
r2 = r2_score(y_test8, y_pred8)

'''35C02'''
X_train = np.concatenate([feature[:760], feature[1423:1722], feature[2021:2320]], axis=0).reshape(-1, 1)
y_train = np.concatenate([cap_df[:760], cap_df[1423:1722], cap_df[2021:2320]], axis=0)
X_test = feature[1722:2021]
y_test2 = cap_df[1722:2021]

model = XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100)
model.fit(X_train, y_train)

y_pred2 = model.predict(X_test)

mae = mean_absolute_error(y_test2, y_pred2)
rmse = np.sqrt(mean_squared_error(y_test2, y_pred2))
mape = np.mean(np.abs((y_test2.values.reshape(1,-1)[0] - y_pred2) / y_test2.values)) * 100
r2 = r2_score(y_test2, y_pred2)

'''45C02'''
X_train = np.concatenate([feature[:760], feature[1423:1722], feature[2021:2320]], axis=0).reshape(-1, 1)
y_train = np.concatenate([cap_df[:760], cap_df[1423:1722], cap_df[2021:2320]], axis=0)
X_test = feature[2320:2630]
y_test4 = cap_df[2320:2630]

model = XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100)
model.fit(X_train, y_train)

y_pred4 = model.predict(X_test)

mae = mean_absolute_error(y_test4, y_pred4)
rmse = np.sqrt(mean_squared_error(y_test4, y_pred4))
mape = np.mean(np.abs((y_test4.values.reshape(1,-1)[0] - y_pred4) / y_test4.values)) * 100
r2 = r2_score(y_test4, y_pred4)
