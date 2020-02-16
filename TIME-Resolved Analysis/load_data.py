import scipy.io
import extract_features
import numpy as np

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest_avg.mat')
data_rest_avg = mat['result_wpli_rest_avg']
data_rest_avg.shape
X_rest_avg=extract_features.extract_features(data_rest_avg)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
data_rest_step.shape
X_rest_step=extract_features.extract_features(data_rest_step)
X_rest_step_diff = extract_features.get_difference(X_rest_step)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes_avg.mat')
data_anes_avg = mat['result_wpli_anes_avg']
data_anes_avg.shape
X_anes_avg=extract_features.extract_features(data_anes_avg)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
data_anes_step.shape
X_anes_step=extract_features.extract_features(data_anes_step)
X_anes_step_diff = extract_features.get_difference(X_anes_step)

#mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest6_avg.mat')
#data_rest6_avg = mat['result_wpli_rest6_avg']
#data_rest6_avg.shape
#X_rest6_avg=extract_features.extract_features(data_rest6_avg)

#mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest6_step.mat')
#data_rest6_step = mat['result_wpli_rest6_step']
#data_rest6_step.shape
#X_rest6_step=extract_features.extract_features(data_rest6_step)
#X_rest6_step_diff = extract_features.get_difference(X_rest6_step)


"""
COMBINE DATASETS
        !!!     1 is anesthesized
        !!!     0 is Resting State
"""
X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)
Y_all=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0])),axis=0)

X_all_test= np.concatenate((X_anes_step[1:20,:],X_rest_step[1:20]),axis=0)
Y_all_test=np.concatenate((np.ones(19),np.zeros(19)),axis=0)

#X_all2= np.concatenate((X_anes_step,X_rest_step,X_rest6_step),axis=0)
#Y_all2=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0]),np.full(X_rest6_step.shape[0], 2)),axis=0)


X_all_diff= np.concatenate((X_anes_step_diff,X_rest_step_diff),axis=0)
Y_all_diff=np.concatenate((np.ones(X_anes_step_diff.shape[0]),np.zeros(X_rest_step_diff.shape[0])),axis=0)
