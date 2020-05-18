import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from import_time_data_mean import *
import seaborn as sns
import matplotlib.backends.backend_pdf

data=pd.read_pickle('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/final_wpli_all_Left_10_1.pickle')

pdf = matplotlib.backends.backend_pdf.PdfPages("output_contrast.pdf")

participants=['02','05','09','10','11','12','13','18','19','20','22','99']

for participant in participants:
    data_base=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Base'))[0],:]
    data_anes=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Anes'))[0],:]
    data_reco=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Reco'))[0],:]

    areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO','FF','CC','PP','TT','OO','MEAN']

    corrB = data_base.iloc[:,4:-1].corr()
    corrA = data_anes.iloc[:,4:-1].corr()
    corrR = data_reco.iloc[:,4:-1].corr()

    figure =plt.figure(figsize=(11,3))
    plt.subplot(131)
    sns.heatmap(corrB,vmin=0, vmax=1)
    plt.title('WSAS'+participant+'Baseline')
    plt.subplot(132)
    sns.heatmap(corrA,vmin=0, vmax=1)
    plt.title('WSAS'+participant+'Anesthesia')
    plt.subplot(133)
    sns.heatmap(corrR,vmin=0, vmax=1)
    plt.title('WSAS'+participant+'Recovery')
    pdf.savefig(figure)
    plt.close()

    figure= plt.figure()
    sns.heatmap(corrA-corrB,vmin=-0.2, vmax=0.5)
    plt.title('WSAS'+participant+' Anes-Base correlation contrast')
    pdf.savefig(figure)
    plt.close()


pdf.close()



plt.plot(data_base['FF'])
plt.plot(data_base['PP'])

FP_corr_B=np.correlate(data_base['FF'],data_base['PP'],'same')
FP_corr_A=np.correlate(data_anes['FF'],data_anes['PP'],'same')
FP_corr_R=np.correlate(data_reco['FF'],data_reco['PP'],'same')


plt.plot(FP_corr_B)
plt.plot(FP_corr_A)
plt.plot(FP_corr_R)
plt.legend(['Baseline','Anesthesia','Recovery'])
plt.title('WSAS 20 Autocottelation of the MEAN')