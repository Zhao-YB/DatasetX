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

file_path3 = r'D:\battery-forecasting-main\test\x_eis_fixed_pure_re.csv'
file_path4 = r'D:\\battery-forecasting-main\test\y_eis_fixed.csv'
# Read the x_eis_im.csv file
x_data_re = pd.read_csv(file_path3, header=None).iloc[:1128,:]

# Read the y_eis.csv file
y_data = pd.read_csv(file_path4, header=None).iloc[:1128,:]

# Initialize an empty matrix to store the results
correlation_matrix = np.zeros((x_data_re.shape[1], x_data_re.shape[1]))

# Iterate through each column of x_eis_im
for i in range(x_data_re.shape[1]):
    # Extract the current column as the baseline column
    baseline_column = x_data_re.iloc[:, i]  
    
    # Iterate through each column of x_eis_im after subtracting the current baseline column
    for j in range(x_data_re.shape[1]):
        # Calculate the modified column
        modified_column = x_data_re.iloc[:, j] - baseline_column
        # Calculate the Pearson correlation coefficient
        corr, _ = pearsonr(modified_column, y_data[0])
        # Store the correlation coefficient in the matrix
        correlation_matrix[j, i] = corr

# Convert the matrix to a DataFrame
result_df_re = pd.DataFrame(correlation_matrix, columns=[f'Column {i+1}' for i in range(x_data_re.shape[1])], index=[f'Column {i+1}' for i in range(x_data_re.shape[1])])

max_value_index  = result_df_abs_re.values.argmax()
print(divmod(max_value_index, result_df_abs_re.shape[1]))
print(result_df_abs_re.values.max())