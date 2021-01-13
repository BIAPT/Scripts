import scipy.io
import sys
sys.path.append('dataimport/')
import dataimport.extract_features
import matplotlib.pyplot as plt
#plt.use('Qt5Agg')
import numpy as np
import pandas as pd
import glob
import pickle


datafiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/Raw_250' + "**/*.mat", recursive=True)]
wplifiles = [f for f in glob.glob('data/WSAS_TIME_DATA_250Hz/wPLI_30_10' + "**/*.mat", recursive=True)]


name=['FC_', 'CP_']
frequencies=np.arange(0, 35, 0.5)
names=[]

names.append('Name')
names.append('ID')
names.append('Phase')
names.append('Time')

for n in name:
    for f in frequencies:
        names.append(n+str(f))


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

    wpli=np.zeros((time_steps,2*freq_steps+4)) # +4 for Name, ID, State, Time
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

    F = ['E11', 'E24', 'E124']
    C = ['E36', 'Cz', 'E104']
    P = ['E52', 'E62', 'E92']

    if ID == 'WSAS02':
        F = ['F3', 'Fz', 'F4']
        C = ['C3', 'Cz', 'C4']
        P = ['P3', 'Pz', 'P4']

    for a in range(0,len(recos)):
        channels.append(recos.iloc[a,0][0])
    channels=np.array(channels)

    for t in range(0, time_steps):
        df_wpli.iloc[t, 3] = t

        # Frontal Central Connectivity
        for f in range(0,freq_steps):
            conn, _ = dataimport.extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=F,selection_2=C,name= name, time= t)
            df_wpli.iloc[t, 4+f] = conn

        # Central-Parietal Connectivity
        for f in range(0,freq_steps):
            conn, _ =dataimport.extract_features.extract_single_features(data_step[f][t],channels=channels,selection_1=C,selection_2=P,name= name, time= t)
            df_wpli.iloc[t, freq_steps+4+f] = conn

    df_wpli_final=df_wpli_final.append(df_wpli)

df_wpli_final.columns = names


df_wpli_final.to_pickle('F_C_P_wPLI_30_10_allfrequ.pickle')
    # data=pd.read_pickle('final_wPLI_clustering.pickle')
