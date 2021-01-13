import scipy.io
import sys
sys.path.append('dataimport/')
import extract_features
import matplotlib.pyplot as plt
#plt.use('Qt5Agg')
import numpy as np
import pandas as pd
import glob
import pickle

#FR=['E15','E1','E9','E16','E10','E3','E2','E11','E4','E124','E123','E122','E5','E118','E117','E116']
#RC=['E6','E112','E111','E110','E106','E105','E104','E103','Cz','E80','E87','E93','E55','E79','E98','E86']
#RP=['E92','E97','E78','E62','E85','E77','E91','E72','E96','E76','E84']
#RO=['E75','E83','E90','E95','E82','E89']
#RT=['E121','E114','E115','E109','E102','E108','E101','E100']

#LF=['E15','E32','E22','E16','E18','E23','E26','E11','E19','E24','E27','E33','E12','E20','E28','E34']
#LC=['E6','E13','E29','E35','E7','E30','E36','E41','Cz','E31','E37','E42','E55','E54','E47','E53']
#LP=['E52','E51','E61','E62','E60','E67','E59','E72','E58','E71','E66']
#LO=['E75','E70','E65','E64','E74','E69']
#LT=['E38','E44','E39','E40','E46','E45','E50','E57']


datafiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/Raw_250' + "**/*.mat", recursive=True)]
wplifiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/wPLI_30_10' + "**/*.mat", recursive=True)]

df_wpli_final=pd.DataFrame()

for i in range(0,len(wplifiles)):
    part=wplifiles[i]
    name=part[37:48]

    #load Data
    mat = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_30_10/' + name + '_300wPLI_30_10.mat')
    data = mat['data']  # extract the variable "data" (3 cell array)
    data_step = data[0][0][0]
    data_avg = data[0][1][0]

    freq_steps=data_step.shape[0]
    time_steps=data_step[0].shape[0]

    wpli=np.zeros((time_steps,10*freq_steps+4)) # +3 for ID, State, Time
    df_wpli=pd.DataFrame(wpli)

    State=part[44:48]
    ID=part[37:43]
    df_wpli.iloc[:,0]=name
    df_wpli.iloc[:,1]=ID
    df_wpli.iloc[:,2]=State

    #data_info=data[0][2][0]
    #boundaries=data_info[0][0][:]
    #window_size=data_info[1][0][0]
    #nr_surrogate=data_info[2][0][0]
    #p_value=data_info[3][0][0]
    #step_size=data_info[4][0][0]

    recording = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/Raw_250/'+name+'_300.mat')
    reco=recording['EEG']
    recos=reco['chanlocs'][0][0][0]
    recos=pd.DataFrame(recos)
    channels=[]

    LF=['E15','E32','E22','E16','E18','E23','E26','E11','E19','E24','E27','E33','E12','E20','E28','E34']
    LC=['E6','E13','E29','E35','E7','E30','E36','E41','Cz','E31','E37','E42','E55','E54','E47','E53']
    LP=['E52','E51','E61','E62','E60','E67','E59','E72','E58','E71','E66']
    LO=['E75','E70','E65','E64','E74','E69']
    LT=['E38','E44','E39','E40','E46','E45','E50','E57']

    if ID == 'WSAS02':
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
        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LF,selection_2=LC,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, 4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LF,selection_2=LP,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, f+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LF,selection_2=LO,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (2*f)+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LF,selection_2=LT,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (3*f)+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LT,selection_2=LO,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (4*f)+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LT,selection_2=LC,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (5*f)+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LT,selection_2=LP,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (6*f)+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LP,selection_2=LO,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (7*f)+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LP,selection_2=LC,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (8*f)+4+f] = mean_conn

        for f in range(0,freq_steps):
            conn=extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=LC,selection_2=LO,name= name)
            # row selection 1
            # col selection 2
            mean_conn=np.mean(conn)
            df_wpli.iloc[t, (9*f)+4+f] = mean_conn

    df_wpli_final=df_wpli_final.append(df_wpli)


#np.save("full_wpli_all_left.npy", df_wpli_final,allow_pickle=True)
df_wpli_final.to_pickle('final_wPLI_all_left.pickle')
#data=pd.read_pickle('final_wPLI_clustering.pickle')
