import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np

#set names
areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO']
frequencies=np.arange(00.5,35.5,0.5)
names=[]

for a in areas:
    for f in frequencies:
        name=str(a)+'_'+str(f)
        names.append(name)

data=pd.read_pickle('data/final_wPLI_all_left.pickle')
Y_ID=data.iloc[:,1]

data_chro=data[(Y_ID == 'WSAS13') | (Y_ID == 'WSAS22') | (Y_ID == 'WSAS10') | (Y_ID == 'WSAS18')]
data_reco=data[(Y_ID == 'WSAS19') | (Y_ID == 'WSAS20') | (Y_ID == 'WSAS02') | (Y_ID == 'WSAS09')]

Part_chro=['WSAS13','WSAS22','WSAS10', 'WSAS18']
Part_reco=['WSAS19','WSAS20','WSAS02','WSAS09']

data_reco.insert(0, 'outcome', "1")
data_chro.insert(0, 'outcome', "0")

data=np.row_stack([data_reco,data_chro])
data=pd.DataFrame(data)
X=data.iloc[:,5:]
X=X.astype(float)
X.columns=names
Y_ID=data.iloc[:,2]
Y_St=data.iloc[:,3]
Y_time=data.iloc[:,4]
Y_out=data.iloc[:,0]

data_Base=data[(Y_St == 'Base')]
X_Base=data_Base.iloc[:,5:]
X_Base.columns=names
X_Base=X_Base.astype(float)
Y_ID_Base=data_Base.iloc[:,2]
Y_time_Base=data_Base.iloc[:,4]
Y_out_Base=data_Base.iloc[:,0]

data_Anes=data[(Y_St == 'Anes')]
X_Anes=data_Anes.iloc[:,5:]
X_Anes.columns=names
X_Anes=X_Anes.astype(float)
Y_ID_Anes=data_Anes.iloc[:,2]
Y_time_Anes=data_Anes.iloc[:,4]
Y_out_Anes=data_Anes.iloc[:,0]

data_Reco=data[(Y_St == 'Reco')]
X_Reco=data_Reco.iloc[:,5:]
X_Reco.columns=names
X_Reco=X_Reco.astype(float)
Y_ID_Reco=data_Reco.iloc[:,2]
Y_time_Reco=data_Reco.iloc[:,4]
Y_out_Reco=data_Reco.iloc[:,0]


# which values do never change
import matplotlib.pyplot as plt
zerostd=np.where(np.std(X)==0)[0]
empty=np.zeros(X.shape[1])
empty[zerostd]=1
