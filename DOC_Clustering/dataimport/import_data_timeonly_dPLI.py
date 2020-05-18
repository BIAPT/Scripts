import scipy.io
import matplotlib.pyplot as plt
#plt.use('Qt5Agg')
import numpy as np
import pandas as pd
import glob
import pickle


from dataimport import extract_features_dPLI


datafiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/Raw_250' + "**/*.mat", recursive=True)]
dplifiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/dPLI_10_1' + "**/*.mat", recursive=True)]

df_dpli_final=pd.DataFrame()

for i in range(0,len(dplifiles)):
    part=dplifiles[i]
    name=part[36:47]
    State = part[43:47]
    ID = part[40:42]

    #load Data
    mat = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/dPLI_10_1/' + name +'_300dPLI_10_1.mat')
    data = mat['result_dpli']  # extract the variable "data" (3 cell array)

    freq_steps=1
    time_steps=data.shape[0]

    dpli=np.zeros((time_steps,10*freq_steps+4+5)) # +4 for ID, State, Time, name +5 intra regional
    df_dpli=pd.DataFrame(dpli)


    df_dpli.iloc[:,0]=name
    df_dpli.iloc[:,1]=ID
    df_dpli.iloc[:,2]=State

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
        LF = ['Fp1', 'AF3', 'AF7', 'AFz', 'Fz', 'F1', 'F3', 'F5', 'F7']
        LC = ['Cz', 'C1', 'C3', 'C5', 'FCz', 'FC1', 'FC3', 'FC5']
        LP = ['Pz', 'P1', 'P3', 'P5', 'P7', 'CP1', 'CP3', 'CP5', 'CPz']
        LO = ['POz', 'PO3', 'PO7', 'Oz', 'O1', 'PO9']
        LT = ['FT7', 'FT9', 'T7', 'TP7', 'TP9']


    for i in range(0,len(recos)):
        channels.append(recos.iloc[i,0][0])
    channels=np.array(channels)


    for t in range(0, time_steps):
        df_dpli.iloc[t, 3] = t

       # Frontal Central Connectivity

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LC,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 4] = mean_conn


        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LP,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 5] = mean_conn


        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LO,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 6] = mean_conn

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LT,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 7] = mean_conn

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LO,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 8] = mean_conn

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LC,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 9] = mean_conn

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LP,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 10] = mean_conn

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LP,selection_2=LO,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 11] = mean_conn

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LP,selection_2=LC,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 12] = mean_conn

        conn=extract_features_dPLI.extract_single_features(data[t],channels=channels,selection_1=LC,selection_2=LO,name= name,time=t)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_dpli.iloc[t, 13] = mean_conn


        # INtraregional
        conn = extract_features_dPLI.extract_single_features(data[t], channels=channels, selection_1=LF, selection_2=LF,name=name,time=t)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_dpli.iloc[t, 14] = mean_conn

        conn = extract_features_dPLI.extract_single_features(data[t], channels=channels, selection_1=LC, selection_2=LC,name=name,time=t)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_dpli.iloc[t, 15] = mean_conn

        conn = extract_features_dPLI.extract_single_features(data[t], channels=channels, selection_1=LP, selection_2=LP,name=name,time=t)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_dpli.iloc[t, 16] = mean_conn

        conn = extract_features_dPLI.extract_single_features(data[t], channels=channels, selection_1=LT, selection_2=LT,name=name,time=t)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_dpli.iloc[t, 17] = mean_conn

        conn = extract_features_dPLI.extract_single_features(data[t], channels=channels, selection_1=LO, selection_2=LO,name=name,time=t)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_dpli.iloc[t, 18] = mean_conn

    df_dpli_final=df_dpli_final.append(df_dpli)

names=['Name','ID','Phase','Time','FC','FP','FO','FT','TO','TC','TP','PO','PC','CO', 'FF','CC','PP','TT','OO']
df_dpli_final.columns=names

df_dpli_final.to_pickle('NEW_dPLI_all_10_1_left.pickle')

