import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_pickle('data/final_wPLI_all_left.pickle')
X=data.iloc[:,4:]
Y_ID=data.iloc[:,1]
Y_St=data.iloc[:,2]
Y_time=data.iloc[:,3]

data_chro=data[(Y_ID == 'WSAS13') | (Y_ID == 'WSAS22') | (Y_ID == 'WSAS10') | (Y_ID == 'WSAS18')]
data_reco=data[(Y_ID == 'WSAS19') | (Y_ID == 'WSAS20') | (Y_ID == 'WSAS02') | (Y_ID == 'WSAS09')]

data_reco.insert(0, 'outcome', "1")
data_chro.insert(0, 'outcome', "0")

data=np.row_stack([data_reco,data_chro])
data=pd.DataFrame(data)
X=data.iloc[:,5:]
Y_ID=data.iloc[:,2]
Y_St=data.iloc[:,3]
Y_time=data.iloc[:,4]
Y_out=data.iloc[:,0]

FC=X.iloc[:,0:70]
FP=X.iloc[:,70:140]
FO=X.iloc[:,140:210]
FT=X.iloc[:,210:280]
TO=X.iloc[:,280:350]
TC=X.iloc[:,350:420]
TP=X.iloc[:,420:490]
PO=X.iloc[:,490:560]
PC=X.iloc[:,560:630]
CO=X.iloc[:,630:700]

frequ=np.arange(0,70,1)

fig,a =  plt.subplots(2,5)
plt.setp(a, xticks=[10,20,30,40,50,60,70] , xticklabels=['5','10','15','20','25','30','35'],
        yticks=[0,0.05,0.1,0.15,0.2,0.25])

a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][0].set_title("Frontal-Central")
a[0][0].set_ylabel("wPLI")
a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][1].set_title("Frontal-Parietal")
a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][2].set_title("Frontal-Occipital")
a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][3].set_title("Frontal-Temporal")
a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][4].set_title("Temporal-Occipital")
a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][0].set_ylabel("wPLI")
a[1][0].set_xlabel("Frequency")
a[1][0].set_title("Temporal-Central")
a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][1].set_title("Temporal-Parietal")
a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][2].set_title("Parietal-Occipital")
a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][3].set_title("Parietal-Central")
a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][4].set_title("Central-Occipital")



fig,a =  plt.subplots(2,5)
plt.setp(a, xticks=[10,20,30,40,50,60,70] , xticklabels=['5','10','15','20','25','30','35'],
        yticks=[0,0.05,0.1,0.15,0.2,0.25])

a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Base') & (Y_out=='1')],0))
a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Base') & (Y_out=='0')],0))
a[0][0].set_title("Frontal-Central")
a[0][0].set_ylabel("wPLI")
a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Base') & (Y_out=='1')],0))
a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Base') & (Y_out=='0')],0))
a[0][1].set_title("Frontal-Parietal")
a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Base') & (Y_out=='1')],0))
a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Base') & (Y_out=='0')],0))
a[0][2].set_title("Frontal-Occipital")
a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Base') & (Y_out=='1')],0))
a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Base') & (Y_out=='0')],0))
a[0][3].set_title("Frontal-Temporal")
a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Base') & (Y_out=='1')],0))
a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Base') & (Y_out=='0')],0))
a[0][4].set_title("Temporal-Occipital")
a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Base') & (Y_out=='1')],0))
a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Base') & (Y_out=='0')],0))
a[1][0].set_ylabel("wPLI")
a[1][0].set_xlabel("Frequency")
a[1][0].set_title("Temporal-Central")
a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Base') & (Y_out=='1')],0))
a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Base') & (Y_out=='0')],0))
a[1][1].set_title("Temporal-Parietal")
a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Base') & (Y_out=='1')],0))
a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Base') & (Y_out=='0')],0))
a[1][2].set_title("Parietal-Occipital")
a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Base') & (Y_out=='1')],0))
a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Base') & (Y_out=='0')],0))
a[1][3].set_title("Parietal-Central")
a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Base') & (Y_out=='1')],0))
a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Base') & (Y_out=='0')],0))
a[1][4].set_title("Central-Occipital")


fig,a =  plt.subplots(2,5)
plt.setp(a, xticks=[10,20,30,40,50,60,70] , xticklabels=['5','10','15','20','25','30','35'],
        yticks=[0,0.05,0.1,0.15,0.2,0.25])

a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Reco') & (Y_out=='1')],0))
a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Reco') & (Y_out=='0')],0))
a[0][0].set_title("Frontal-Central")
a[0][0].set_ylabel("wPLI")
a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Reco') & (Y_out=='1')],0))
a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Reco') & (Y_out=='0')],0))
a[0][1].set_title("Frontal-Parietal")
a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Reco') & (Y_out=='1')],0))
a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Reco') & (Y_out=='0')],0))
a[0][2].set_title("Frontal-Occipital")
a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Reco') & (Y_out=='1')],0))
a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Reco') & (Y_out=='0')],0))
a[0][3].set_title("Frontal-Temporal")
a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Reco') & (Y_out=='1')],0))
a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Reco') & (Y_out=='0')],0))
a[0][4].set_title("Temporal-Occipital")
a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Reco') & (Y_out=='1')],0))
a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Reco') & (Y_out=='0')],0))
a[1][0].set_ylabel("wPLI")
a[1][0].set_xlabel("Frequency")
a[1][0].set_title("Temporal-Central")
a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Reco') & (Y_out=='1')],0))
a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Reco') & (Y_out=='0')],0))
a[1][1].set_title("Temporal-Parietal")
a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Reco') & (Y_out=='1')],0))
a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Reco') & (Y_out=='0')],0))
a[1][2].set_title("Parietal-Occipital")
a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Reco') & (Y_out=='1')],0))
a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Reco') & (Y_out=='0')],0))
a[1][3].set_title("Parietal-Central")
a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Reco') & (Y_out=='1')],0))
a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Reco') & (Y_out=='0')],0))
a[1][4].set_title("Central-Occipital")

