import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from import_time_data_mean import *
import seaborn as sns
import matplotlib.backends.backend_pdf

data=pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
areas=data.columns[4:]

pdf = matplotlib.backends.backend_pdf.PdfPages("output_contrast_wholebrain.pdf")

participants=['02','05','09','10','11','12','13','18','19','20','22']

for participant in participants:
    data_base=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Base'))[0],:]
    data_anes=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Anes'))[0],:]
    data_reco=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Reco'))[0],:]

    corrB = data_base.iloc[:,4:].corr()
    corrA = data_anes.iloc[:,4:].corr()
    corrR = data_reco.iloc[:,4:].corr()

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
