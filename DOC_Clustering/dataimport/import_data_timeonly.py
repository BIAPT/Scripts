import scipy.io
import sys
import matplotlib.pyplot as plt
#plt.use('Qt5Agg')
import numpy as np
import pandas as pd
import glob
import pickle

from dataimport import extract_features

datafiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/Raw_250' + "**/*.mat", recursive=True)]
wplifiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/wPLI_10_1_alpha' + "**/*.mat", recursive=True)]

df_wpli_final=pd.DataFrame()


for i in range(0,len(wplifiles)):
    part=wplifiles[i]
    name=part[42:53]
    State = part[49:53]
    ID = part[46:48]

    #load Data
    mat = scipy.io.loadmat(wplifiles[i])
    data = mat['result_wpli']  # extract the variable "data" (3 cell array)

    freq_steps=1
    time_steps=data.shape[0]

    wpli=np.zeros((time_steps,10*freq_steps+5+4)) # +4 for ID, State, Time, name +5 intra regional
    df_wpli=pd.DataFrame(wpli)


    df_wpli.iloc[:,0]=name
    df_wpli.iloc[:,1]=ID
    df_wpli.iloc[:,2]=State

    #data_info=data[0][2][0]
    #boundaries=data_info[0][0][:]
    #window_size=data_info[1][0][0]
    #nr_surrogate=data_info[2][0][0]
    #p_value=data_info[3][0][0]
    #step_size=data_info[4][0][0]

    recording = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/Raw_250/'+ 'WSAS'+ID+'_'+State + '_300.mat')
    reco=recording['EEG']
    recos=reco['chanlocs'][0][0][0]
    recos=pd.DataFrame(recos)
    channels=[]

    LF = ['E15', 'E32', 'E22', 'E16', 'E18', 'E23', 'E26', 'E11', 'E19', 'E24', 'E27', 'E33', 'E12', 'E20', 'E28', 'E34']
    LC = ['E6', 'E13', 'E29', 'E35', 'E7', 'E30', 'E36', 'E41', 'Cz', 'E31', 'E37', 'E42', 'E55', 'E54', 'E47', 'E53']
    LP = ['E52', 'E51', 'E61', 'E62', 'E60', 'E67', 'E59', 'E72', 'E58', 'E71', 'E66']
    LO = ['E75', 'E70', 'E65', 'E64', 'E74', 'E69']
    LT = ['E38', 'E44', 'E39', 'E40', 'E46', 'E45', 'E50', 'E57']

    if ID == '02':
        LF=['Fp1','AF3','AF7','AFz','Fz','F1','F3','F5','F7']
        LC=['Cz','C1','C3','C5','FCz','FC1','FC3','FC5']
        LP=['Pz','P1','P3','P5','P7','CP1','CP3','CP5','CPz']
        LO=['POz','PO3','PO7','Oz','O1','PO9']
        LT=['FT7','FT9','T7','TP7','TP9']


    for i in range(0,len(recos)):
        channels.append(recos.iloc[i,0][0])
    channels=np.array(channels)


    for t in range(0, time_steps):
        df_wpli.iloc[t, 3] = t

       # Frontal Central Connectivity

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LC,name= name,time=t)
        df_wpli.iloc[t, 4] = conn


        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LP,name= name,time=t+1)
        df_wpli.iloc[t, 5] = conn


        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LO,name= name,time=t+1)
        df_wpli.iloc[t, 6] = conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LT,name= name,time=t+1)
        df_wpli.iloc[t, 7] = conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LO,name= name,time=t+1)
        df_wpli.iloc[t, 8] = conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LC,name= name,time=t+1)
        df_wpli.iloc[t, 9] = conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LP,name= name,time=t+1)
        df_wpli.iloc[t, 10] = conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LP,selection_2=LO,name= name,time=t+1)
        df_wpli.iloc[t, 11] = conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LP,selection_2=LC,name= name,time=t+1)
        df_wpli.iloc[t, 12] = conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LC,selection_2=LO,name= name,time=t+1)
        df_wpli.iloc[t, 13] = conn

        # INtraregional
        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LF, selection_2=LF,name=name,time=t+1)
        df_wpli.iloc[t, 14] = conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LC, selection_2=LC,name=name,time=t+1)
        df_wpli.iloc[t, 15] = conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LP, selection_2=LP,name=name,time=t+1)
        df_wpli.iloc[t, 16] = conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LT, selection_2=LT,name=name,time=t+1)
        df_wpli.iloc[t, 17] = conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LO, selection_2=LO,name=name,time=t+1)
        df_wpli.iloc[t, 18] = conn

    df_wpli_final=df_wpli_final.append(df_wpli)

names=['Name','ID','Phase','Time','FC','FP','FO','FT','TO','TC','TP','PO','PC','CO', 'FF','CC','PP','TT','OO']
df_wpli_final.columns=names

df_wpli_final.to_pickle('NEW_wPLI_all_10_1_left_alpha.pickle')

