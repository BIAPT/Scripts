import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prepareDataset import *

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

plt.imshow(np.transpose(FC.iloc[1:20,:]))
plt.ylabel('frequency [Hz]')
plt.xlabel('timestep')
plt.colorbar()

eFC=empty[0:70]
eFP=empty[70:140]
eFO=empty[140:210]
eFT=empty[210:280]
eTO=empty[280:350]
eTC=empty[350:420]
eTP=empty[420:490]
ePO=empty[490:560]
ePC=empty[560:630]
eCO=empty[630:700]

frequ=np.arange(0,70,1)

fig,a =plt.subplots(2,5)
plt.setp(a, xticks=[10,20,30,40,50,60,70] , xticklabels=['5','10','15','20','25','30','35'])

a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][0].plot(frequ,np.mean(FC[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][0].set_title("Frontal-Central")
a[0][0].set_ylabel("wPLI")
xcoords = frequ[eFC==1]
for xc in xcoords:
        a[0][0].axvline(x=xc,color='red',alpha=0.2)

a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][1].plot(frequ,np.mean(FP[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][1].set_title("Frontal-Parietal")
xcoords = frequ[eFP==1]
for xc in xcoords:
        a[0][1].axvline(x=xc,color='red',alpha=0.2)

a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][2].plot(frequ,np.mean(FO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][2].set_title("Frontal-Occipital")
xcoords = frequ[eFO==1]
for xc in xcoords:
        a[0][2].axvline(x=xc,color='red',alpha=0.2)

a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][3].plot(frequ,np.mean(FT[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][3].set_title("Frontal-Temporal")
xcoords = frequ[eFT==1]
for xc in xcoords:
        a[0][3].axvline(x=xc,color='red',alpha=0.2)

a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[0][4].plot(frequ,np.mean(TO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[0][4].set_title("Temporal-Occipital")
xcoords = frequ[eTO==1]
for xc in xcoords:
        a[0][4].axvline(x=xc,color='red',alpha=0.2)

a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][0].plot(frequ,np.mean(TC[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][0].set_ylabel("wPLI")
a[1][0].set_xlabel("Frequency")
a[1][0].set_title("Temporal-Central")
xcoords = frequ[eTC==1]
for xc in xcoords:
        a[1][0].axvline(x=xc,color='red',alpha=0.2)

a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][1].plot(frequ,np.mean(TP[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][1].set_title("Temporal-Parietal")
xcoords = frequ[eTP==1]
for xc in xcoords:
        a[1][1].axvline(x=xc,color='red',alpha=0.2)

a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][2].plot(frequ,np.mean(PO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][2].set_title("Parietal-Occipital")
xcoords = frequ[ePO==1]
for xc in xcoords:
        a[1][2].axvline(x=xc,color='red',alpha=0.2)

a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][3].plot(frequ,np.mean(PC[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][3].set_title("Parietal-Central")
xcoords = frequ[ePC==1]
for xc in xcoords:
        a[1][3].axvline(x=xc,color='red',alpha=0.2)

a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Anes') & (Y_out=='1')],0))
a[1][4].plot(frequ,np.mean(CO[(Y_St == 'Anes') & (Y_out=='0')],0))
a[1][4].set_title("Central-Occipital")
xcoords = frequ[eCO==1]
for xc in xcoords:
        a[1][4].axvline(x=xc,color='red',alpha=0.2)



fig,a =  plt.subplots(2,5)
plt.setp(a, xticks=[10,20,30,40,50,60,70] , xticklabels=['5','10','15','20','25','30','35'])

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
xcoords = frequ[eFC==1]
for xc in xcoords:
        a[0][0].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eFP==1]
for xc in xcoords:
        a[0][1].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eFO==1]
for xc in xcoords:
        a[0][2].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eFT==1]
for xc in xcoords:
        a[0][3].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eTO==1]
for xc in xcoords:
        a[0][4].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eTC==1]
for xc in xcoords:
        a[1][0].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eTP==1]
for xc in xcoords:
        a[1][1].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[ePO==1]
for xc in xcoords:
        a[1][2].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[ePC==1]
for xc in xcoords:
        a[1][3].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eCO==1]
for xc in xcoords:
        a[1][4].axvline(x=xc,color='red',alpha=0.2)


fig,a =  plt.subplots(2,5)
plt.setp(a, xticks=[10,20,30,40,50,60,70] , xticklabels=['5','10','15','20','25','30','35'])

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
xcoords = frequ[eFC==1]
for xc in xcoords:
        a[0][0].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eFP==1]
for xc in xcoords:
        a[0][1].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eFO==1]
for xc in xcoords:
        a[0][2].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eFT==1]
for xc in xcoords:
        a[0][3].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eTO==1]
for xc in xcoords:
        a[0][4].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eTC==1]
for xc in xcoords:
        a[1][0].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eTP==1]
for xc in xcoords:
        a[1][1].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[ePO==1]
for xc in xcoords:
        a[1][2].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[ePC==1]
for xc in xcoords:
        a[1][3].axvline(x=xc,color='red',alpha=0.2)
xcoords = frequ[eCO==1]
for xc in xcoords:
        a[1][4].axvline(x=xc,color='red',alpha=0.2)



