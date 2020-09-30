import scipy.io
import numpy as np
import sys
import os
import pandas as pd
sys.path.append('../')
from helper_functions import extract_features

datadir = 'data/BASELINE_5min_250Hz/'
datafiles = [os.path.join(datadir, f) for f in os.listdir(datadir) if '.mat' in f]
wplidir = 'data/wPLI_10_10_alpha/'
wplifiles = [os.path.join(wplidir, f) for f in os.listdir(wplidir) if '.mat' in f]

df_wpli_final=pd.DataFrame()

ROI = ['LF_LC', 'LF_LP', 'LF_LO', 'LF_LT', 'LT_LO', 'LT_LC', 'LT_LP', 'LP_LO', 'LP_LC', 'LC_LO', 'LF_LF', 'LC_LC',
           'LP_LP', 'LT_LT', 'LO_LO',
           'RF_RC', 'RF_RP', 'RF_RO', 'RF_RT', 'RT_RO', 'RT_RC', 'RT_RP', 'RP_RO', 'RP_RC', 'RC_RO', 'RF_RF', 'RC_RC',
           'RP_RP', 'RT_RT', 'RO_RO',
           'LF_RC', 'LF_RP', 'LF_RO', 'LF_RT', 'LT_RO', 'LT_RC', 'LT_RP', 'LP_RO', 'LP_RC', 'LC_RO',
           'RF_LC', 'RF_LP', 'RF_LO', 'RF_LT', 'RT_LO', 'RT_LC', 'RT_LP', 'RP_LO', 'RP_LC', 'RC_LO',
           'LF_RF', 'LC_RC', 'LP_RP', 'LT_RT', 'LO_RO']

names = ROI.copy()
names.insert(0, 'Name')
names.insert(1, 'ID')
names.insert(2, 'Phase')
names.insert(3, 'Time')


for i in range(0,len(wplifiles)):
    part = wplifiles[i]
    name = part[-31:-20]
    State = part[-24:-20]
    ID = part[-28:-25]

    #load Data
    mat = scipy.io.loadmat(wplifiles[i])
    data = mat['result_wpli'] # extract the variable "data" (3 cell array)
    missingel = []
    time_steps=data.shape[0]

    wpli=np.zeros((time_steps,len(names))) # +4 for ID, State, Time, name +5 intra regional
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


    recording = scipy.io.loadmat( datadir + name + '_5min.mat')
    reco=recording['EEG']
    recos=reco['chanlocs'][0][0][0]
    recos=pd.DataFrame(recos)


    channels=[]

    # initialize a dict of regions and referring electrodes
    regions = {}

    regions["LF"] = ['E15', 'E32', 'E22', 'E16', 'E18', 'E23', 'E26', 'E11', 'E19', 'E24', 'E27', 'E33', 'E12', 'E20', 'E28', 'E34']
    regions["LC"] = ['E6', 'E13', 'E29', 'E35', 'E7', 'E30', 'E36', 'E41', 'Cz', 'E31', 'E37', 'E42', 'E55', 'E54', 'E47', 'E53']
    regions["LP"] = ['E52', 'E51', 'E61', 'E62', 'E60', 'E67', 'E59', 'E72', 'E58', 'E71', 'E66']
    regions["LO"] = ['E75', 'E70', 'E65', 'E64', 'E74', 'E69']
    regions["LT"] = ['E38', 'E44', 'E39', 'E40', 'E46', 'E45', 'E50', 'E57']

    regions["RF"] = ['E15', 'E1', 'E9', 'E16', 'E10', 'E3', 'E2', 'E11', 'E4', 'E124', 'E123', 'E122', 'E5', 'E118', 'E117', 'E116']
    regions["RC"] = ['E6', 'E112', 'E111', 'E110', 'E106', 'E105', 'E104', 'E103', 'Cz', 'E80', 'E87', 'E93', 'E55', 'E79', 'E98', 'E86']
    regions["RP"] = ['E92', 'E97', 'E78', 'E62', 'E85', 'E77', 'E91', 'E72', 'E96', 'E76', 'E84']
    regions["RO"] = ['E75', 'E83', 'E90', 'E95', 'E82', 'E89']
    regions["RT"] = ['E121', 'E114', 'E115', 'E109', 'E102', 'E108', 'E101', 'E100']

    if ID.__contains__('A') and ID != ('A17') :
        regions["LF"] = ['E15', 'E32', 'Fp1', 'E16', 'E18', 'E23', 'E26', 'Fz', 'E19', 'F3', 'E27', 'F7', 'E12',
                         'E20', 'E28', 'E34']
        regions["LC"] = ['E6', 'E13', 'E29', 'E35', 'E7', 'E30', 'C3', 'E41', 'Cz', 'E31', 'E37', 'E42', 'E55', 'E54',
                         'E47', 'E53']
        regions["LP"] = ['P3', 'E51', 'E61', 'Pz', 'E60', 'E67', 'E59', 'E72', 'P7', 'E71', 'E66']
        regions["LO"] = ['Oz', 'O1', 'E65', 'E64', 'E74', 'E69']
        regions["LT"] = ['E38', 'E44', 'E39', 'E40', 'E46', 'T7', 'E50', 'LM']

        regions["RF"] = ['E15', 'E1', 'Fp2', 'E16', 'E10', 'E3', 'E2', 'Fz', 'E4', 'F4', 'E123', 'F8', 'E5', 'E118',
                         'E117', 'E116']
        regions["RC"] = ['E6', 'E112', 'E111', 'E110', 'E106', 'E105', 'C4', 'E103', 'Cz', 'E80', 'E87', 'E93', 'E55',
                         'E79', 'E98', 'E86']
        regions["RP"] = ['P4', 'E97', 'E78', 'Pz', 'E85', 'E77', 'E91', 'E72', 'P8', 'E76', 'E84']
        regions["RO"] = ['Oz', 'O2', 'E90', 'E95', 'E82', 'E89']
        regions["RT"] = ['E121', 'E114', 'E115', 'E109', 'E102', 'T8', 'E101', 'RM']

    if ID == 'S02':

        regions = {}
        regions["LF"] = ['Fp1', 'AF3', 'AF7', 'AFz', 'Fz', 'F1', 'F3', 'F5', 'F7']
        regions["LC"] = ['Cz', 'C1', 'C3', 'C5', 'FCz', 'FC1', 'FC3', 'FC5']
        regions["LP"] = ['Pz', 'P1','P3', 'P5', 'P7', 'CP1', 'CP3', 'CP5', 'CPz']
        regions["LO"] = ['POz', 'PO3', 'PO7', 'Oz', 'O1', 'PO9']
        regions["LT"] = ['FT7', 'FT9', 'T7', 'TP7', 'TP9']

        regions['RF'] = ['Fp2', 'AF4', 'AF8', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8']
        regions['RC'] = ['Cz', 'C2', 'C4', 'C6', 'FCz', 'FC2', 'FC4', 'FC6']
        regions['RP'] = ['Pz', 'P2', 'P4', 'P6', 'P8', 'CP2', 'CP4', 'CP6', 'CPz']
        regions['RO'] = ['POz', 'PO4', 'PO8', 'Oz', 'O2', 'PO10']
        regions['RT'] = ['FT8', 'FT10', 'T8', 'TP8', 'TP10']

    for a in range(0,len(recos)):
        channels.append(recos.iloc[a,0][0])
    channels=np.array(channels)

    for t in range(0, time_steps):
        df_wpli.iloc[t, 3] = t
        i = 4   # Position in DataFrame: 0-3 are 'Name','ID','Phase','Time'

        for r in ROI:
            r1=r[0:2]
            r2=r[3:5]

            conn,missing=extract_features.extract_single_features(data[t],channels=channels,
                                                          selection_1=regions[r1],selection_2=regions[r2],
                                                          name= name,time=t)
            missingel.extend(missing)
            df_wpli.iloc[t, i] = conn

            i += 1

    df_wpli_final=df_wpli_final.append(df_wpli)
    print( "Participant" + name + "   finished")
    print("missing electrodes: " + str(list(set(missingel))))

df_wpli_final.columns = names

df_wpli_final.to_pickle('33_Part_WholeBrain_wPLI_10_10_alpha.pickle', protocol=4)

