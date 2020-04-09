import scipy.io
import extract_features
import matplotlib.pyplot as plt
#plt.use('Qt5Agg')
import numpy as np
import pandas as pd
import glob
import pickle


# import all part and states Mean
mean_12_base = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/wpli_12_Base.mat')
data_12_base = mean_12_base['result_wpli_12base_step']
mean_12_base = np.mean(data_12_base,axis=0)

mean_12_reco = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/wpli_12_Reco.mat')
data_12_reco = mean_12_reco['result_wpli_12reco_step']
mean_12_reco = np.mean(data_12_reco,axis=0)

mean_12_anes = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/wpli_12_Anes.mat')
data_12_anes = mean_12_anes['result_wpli_12anes_step']
mean_12_anes = np.mean(data_12_anes,axis=0)

# import all part and states Mean
mean_20_base = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/wpli_20_Base.mat')
data_20_base = mean_20_base['result_wpli_20base_step']
mean_20_base = np.mean(data_20_base,axis=0)

mean_20_reco = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/wpli_20_Reco.mat')
data_20_reco = mean_20_reco['result_wpli_20reco_step']
mean_20_reco = np.mean(data_20_reco,axis=0)

mean_20_anes = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/wpli_20_Anes.mat')
data_20_anes = mean_20_anes['result_wpli_20anes_step']
mean_20_anes = np.mean(data_20_anes,axis=0)

