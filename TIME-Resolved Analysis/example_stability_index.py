import scipy.io
import extract_features
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import get_stability_index
from matplotlib import pyplot as plt


mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
X_anes_step=extract_features.extract_features(data_anes_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
X_rest_step=extract_features.extract_features(data_rest_step)

X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)

"""
Stability Index
"""
P=[2,3,4,5]     #number of Principal components to iterate
K=[2,3,4,5]     #number of K-clusters to iterate
Rep=2          #number of Repetitions (Mean at the end)

X_temp=X_rest_step    #Template set (50% of Participants)
X_test=X_anes_step    #Test set (50% of Participants)

[SI_M ,SI_SD] = get_stability_index.Stability_Index(X_temp,X_test,P,K,Rep)


