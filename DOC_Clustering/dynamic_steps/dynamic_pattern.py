import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
import numpy as np
import matplotlib.backends.backend_pdf

part = ['13', '18', '05', '11', '19', '02', '20', '22', '12', '10', '09']
pdf = matplotlib.backends.backend_pdf.PdfPages("wholebrain_Pattern.pdf")
data = pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
areas = data.columns[4:]

for p in part:
    data_p=data.query("ID == '{}'".format(p))
    data_p_Base=data_p.query("Phase == 'Base'")
    data_p_Anes=data_p.query("Phase == 'Anes'")
    data_p_Reco=data_p.query("Phase == 'Reco'")

    max_Base=np.zeros(data_p_Base[areas].shape)
    max_Anes=np.zeros(data_p_Anes[areas].shape)

    for i in range(len(data_p_Base)):
        maxarea=np.where(data_p_Base[areas].iloc[i,:]==max(data_p_Base[areas].iloc[i,:]))[0][0]
        max_Base[i,maxarea]=1

    for i in range(len(data_p_Anes)):
        maxarea=np.where(data_p_Anes[areas].iloc[i,:]==max(data_p_Anes[areas].iloc[i,:]))[0][0]
        max_Anes[i,maxarea]=1

    fig, ax= plt.subplots(2,1)
    fig.suptitle('Brain_Melody Part: {}'.format(p))
    ax[0].imshow(np.transpose(max_Base))
    ax[1].imshow(np.transpose(max_Anes))
    pdf.savefig(fig)
    plt.close()

pdf.close()

