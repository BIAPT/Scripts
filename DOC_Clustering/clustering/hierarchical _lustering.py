import matplotlib
matplotlib.use('Qt5Agg')
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy import stats

data=pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
Phase='Anesthesia'

data = data.query("Phase=='Base'")
areas=data.columns[4:]
Part=['13','22','10', '18','05','12','11','19','20','02','09']

max_len=190
time_data=np.zeros([len(Part),max_len,data.shape[1]-4])
ID=Part

for p in range(len(Part)):
    tmp = data.query("ID=='{}'".format(Part[p]))
    time_data[p]=tmp.iloc[:max_len,4:]

def KScoeff(df):
    ks_matrix = np.zeros((len(df), len(df)))
    ps_matrix = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            ks_test = stats.ks_2samp(df[i], df[j])
            ks_matrix[i, j] = ks_test.statistic
            ks_matrix[j, i] = ks_test.statistic
            ps_matrix[i, j] = ks_test.pvalue
            ps_matrix[j, i] = ks_test.pvalue

    return ks_matrix, ps_matrix

### COMPUTE KOLMOGOROV SMIRNOV MATRIX ###

import matplotlib.backends.backend_pdf

name = "Hierarchical_clustering_dPLI_"+Phase+".pdf"
pdf = matplotlib.backends.backend_pdf.PdfPages(name)

for i in range(len(areas)):
    df = time_data[:]
    df=df[:,:,i]
    ks_matrix,ps_matrix = KScoeff(df)

    ### HIERACHICAL CLUSTERING ###
    d = sch.distance.pdist(ks_matrix)
    L = sch.linkage(d, method='ward')
    ind = sch.fcluster(L, d.max(), 'distance')
    dendrogram = sch.dendrogram(L, no_plot=True)

    df = [df[i] for i in dendrogram['leaves']]
    labels = [ID[:][i] for i in dendrogram['leaves']]
    ks_matrix,kp_matrix = KScoeff(df)

    Figure=plt.figure(figsize=(6,6))
    plt.imshow(ks_matrix)
    plt.xticks(range(len(ID[:])), labels)
    plt.yticks(range(len(ID[:])), labels)
    np.set_printoptions(False)
    pdf.savefig(Figure)
    plt.close()

    ### PLOT DENDROGRAM ###

    Figure=plt.figure(figsize=(8,6))
    dendrogram = sch.dendrogram(L, labels=ID[:])
    plt.axhline(d.max(), c='black')
    plt.title(Phase + '  '+areas[i])
    pdf.savefig(Figure)
    plt.close()

pdf.close()


