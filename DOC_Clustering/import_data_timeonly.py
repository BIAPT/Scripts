import scipy.io
import extract_features
import matplotlib.pyplot as plt
#plt.use('Qt5Agg')
import numpy as np
import pandas as pd
import glob
import pickle


datafiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/Raw_250' + "**/*.mat", recursive=True)]
wplifiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/wPLI_10_1' + "**/*.mat", recursive=True)]


df_wpli_final=pd.DataFrame()



for i in range(0,len(wplifiles)):
    part=wplifiles[i]
    name=part[36:47]
    State = part[43:47]
    ID = part[40:42]

    #load Data
    mat = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/' + name +'_300wPLI_10_1.mat')
    data = mat['result_wpli']  # extract the variable "data" (3 cell array)

    freq_steps=1
    time_steps=data.shape[0]

    wpli=np.zeros((time_steps,10*freq_steps+5+5)) # +5 for ID, State, Time, name, average +5 intra regional
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
        LF=['Fp1','AF3','AF7','AFz','Fz','F1','F3','F5','F7','FCz','FC1','FC3','FC5','FT7','FT9']
        LC=['Cz','C1','C3','C5','CPz','CP1','CP3','CP5']
        LP=['Pz','P1','P3','P5','P7','POz','PO3','PO7']
        LO=['Oz','O1']
        LT=['T7','TP7','TP9']


    for i in range(0,len(recos)):
        channels.append(recos.iloc[i,0][0])
    channels=np.array(channels)


    for t in range(0, time_steps):
        df_wpli.iloc[t, 3] = t

       # Frontal Central Connectivity

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LC,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 4] = mean_conn


        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LP,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 5] = mean_conn


        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LO,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 6] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LF,selection_2=LT,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 7] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LO,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 8] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LC,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 9] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LT,selection_2=LP,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 10] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LP,selection_2=LO,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 11] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LP,selection_2=LC,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 12] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=LC,selection_2=LO,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 13] = mean_conn


        # INtraregional
        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LF, selection_2=LF,name=name)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_wpli.iloc[t, 14] = mean_conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LC, selection_2=LC,name=name)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_wpli.iloc[t, 15] = mean_conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LP, selection_2=LP,name=name)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_wpli.iloc[t, 16] = mean_conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LT, selection_2=LT,name=name)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_wpli.iloc[t, 17] = mean_conn

        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=LO, selection_2=LO,name=name)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_wpli.iloc[t, 18] = mean_conn


        # AVERAGE CONNECTIVITY
        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=channels, selection_2=channels, name=name)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_wpli.iloc[t, 19] = mean_conn



    df_wpli_final=df_wpli_final.append(df_wpli)

names=['Name','ID','Phase','Time','FC','FP','FO','FT','TO','TC','TP','PO','PC','CO', 'FF','CC','PP','TT','OO','MEAN']
df_wpli_final.columns=names

#np.save("time_resolved_wpli_all.npy", df_wpli_final,allow_pickle=True)
df_wpli_final.to_pickle('final_wPLI_all_10_1_allWSAS_05MDFA.pickle')
#data=pd.read_pickle('final_wPLI_clustering.pickle')

