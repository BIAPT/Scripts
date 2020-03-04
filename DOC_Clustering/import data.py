import scipy.io
import extract_features
import matplotlib
#matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd

mat = scipy.io.loadmat('data/WSAS22_Rec_wPLI.mat')
data = mat['data']  #extract the variable "data" (3 cell array)
data_step=data[0][0][0]
data_avg=data[0][1][0]
data_info=data[0][2][0]

boundaries=data_info[0][0][:]
window_size=data_info[1][0][0]
nr_surrogate=data_info[2][0][0]
p_value=data_info[3][0][0]
step_size=data_info[4][0][0]

recording = scipy.io.loadmat('data/WSAS22_Rec_300.mat')
reco=recording['EEG']
recos=reco['chanlocs'][0][0][0]
recos=pd.DataFrame(recos)
channels=[]
for i in range(0,len(recos)):
    channels.append(recos.iloc[i,0][0])
channels=np.array(channels)
len(channels)

selection1 = ['E11', 'E24', 'E124']
selection2 = ['E36', 'Cz', 'E104']

test=extract_features.extract_single_features(data_avg[0],channels=channels,selection_1=selection1,selection_2=selection2)
# row selection 1
# col selection 2

fc_mean=np.mean(test)
