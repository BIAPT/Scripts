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


def choose_maximum(data):
    max_frame = np.zeros(data.shape)

    for i in range(len(data)):
        maxarea = np.where(data.iloc[i, :] == max(data.iloc[i, :]))[0][0]
        max_frame[i, maxarea] = 1

    return max_frame

def choose_range(data):
    max_frame = np.zeros(data.shape)

    for i in range(len(data)):
        timestep = data.iloc[i, :].copy()

        for a in range(data.shape[1]):
            maxarea = np.where(timestep == max(timestep))[0][0]
            max_frame[i, maxarea] = a
            # set maxarea to small value
            timestep[maxarea] = -100
            a +=1

    return max_frame


for p in part:
    data_p=data.query("ID == '{}'".format(p))
    data_p_Base=data_p.query("Phase == 'Base'")[areas]
    data_p_Anes=data_p.query("Phase == 'Anes'")[areas]
    data_p_Reco=data_p.query("Phase == 'Reco'")[areas]

    max_Base = choose_maximum(data_p_Base)
    max_Anes = choose_maximum(data_p_Anes)

    range_Base = choose_range(data_p_Base)
    range_Anes = choose_range(data_p_Anes)

    fig, ax= plt.subplots(2,1)
    fig.suptitle('Brain_Melody Part: {}'.format(p))
    ax[0].imshow(np.transpose(max_Base))
    ax[1].imshow(np.transpose(max_Anes))
    pdf.savefig(fig)
    plt.close()

    fig, ax= plt.subplots(2,1)
    fig.suptitle('Brain_Melody Part: {}'.format(p))
    ax[0].imshow(np.transpose(range_Base),cmap = 'jet' )
    ax[1].imshow(np.transpose(range_Anes),cmap = 'jet')
    pdf.savefig(fig)
    plt.close()

pdf.close()

import scipy.cluster.hierarchy as sch

### HIERACHICAL CLUSTERING ###
d = sch.distance.pdist(range_Anes)
L = sch.linkage(d, method='ward')
ind = sch.fcluster(L, d.max(), 'distance')
dendrogram = sch.dendrogram(L, no_plot=True)

plt.figure()
df = [np.array(range_Anes)[i] for i in dendrogram['leaves']]
plt.imshow(np.transpose(df),cmap ='jet')

Figure = plt.figure(figsize=(8, 6))
dendrogram = sch.dendrogram(L)
plt.axhline(d.max(), c='black')
