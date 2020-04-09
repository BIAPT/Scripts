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
    name=part[36:48]
    State = part[44:48]
    ID = part[41:43]

    #load Data
    mat = scipy.io.loadmat('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/' + name + '.mat')
    data = mat['result_wpli_'+ID+ State.lower()+'_step']  # extract the variable "data" (3 cell array)

    freq_steps=1
    time_steps=data.shape[0]

    wpli=np.zeros((time_steps,10*freq_steps+5)) # +5 for ID, State, Time, name, average
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



    F=list(set(['E15','E32','E22','E16','E18','E23','E26','E11','E19','E24','E27','E33','E12','E20','E28','E34','E15','E1','E9','E16','E10','E3','E2','E11','E4','E124','E123','E122','E5','E118','E117','E116']))
    C=list(set(['E6','E13','E29','E35','E7','E30','E36','E41','Cz','E31','E37','E42','E55','E54','E47','E53','E6','E112','E111','E110','E106','E105','E104','E103','Cz','E80','E87','E93','E55','E79','E98','E86']))
    P=list(set(['E52','E51','E61','E62','E60','E67','E59','E72','E58','E71','E66','E92','E97','E78','E62','E85','E77','E91','E72','E96','E76','E84']))
    O=list(set(['E75','E70','E65','E64','E74','E69','E75','E83','E90','E95','E82','E89']))
    T=list(set(['E38','E44','E39','E40','E46','E45','E50','E57','E121','E114','E115','E109','E102','E108','E101','E100']))


    for i in range(0,len(recos)):
        channels.append(recos.iloc[i,0][0])
    channels=np.array(channels)


    for t in range(0, time_steps):
        df_wpli.iloc[t, 3] = t

       # Frontal Central Connectivity

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=F,selection_2=C,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 4] = mean_conn


        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=F,selection_2=P,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 5] = mean_conn


        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=F,selection_2=O,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 6] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=F,selection_2=T,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 7] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=T,selection_2=O,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 8] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=T,selection_2=C,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 9] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=T,selection_2=P,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 10] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=P,selection_2=O,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 11] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=P,selection_2=C,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 12] = mean_conn

        conn=extract_features.extract_single_features(data[t],channels=channels,selection_1=C,selection_2=O,name= name)
        # row selection 1
        # col selection 2
        mean_conn=np.mean(conn)
        df_wpli.iloc[t, 13] = mean_conn


        # AVERAGE CONNECTIVITY
        conn = extract_features.extract_single_features(data[t], channels=channels, selection_1=channels, selection_2=channels,name=name)
        # row selection 1
        # col selection 2
        mean_conn = np.mean(conn)
        df_wpli.iloc[t, 14] = mean_conn

    df_wpli_final=df_wpli_final.append(df_wpli)

names=['Name','ID','Phase','Time','FC','FP','FO','FT','TO','TC','TP','PO','PC','CO','MEAN']
df_wpli_final.columns=names

#np.save("time_resolved_wpli_all.npy", df_wpli_final,allow_pickle=True)
df_wpli_final.to_pickle('final_wPLI_all_NEW.pickle')
#data=pd.read_pickle('final_wPLI_clustering.pickle')
