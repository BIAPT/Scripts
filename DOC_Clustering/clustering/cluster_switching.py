import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf

data=pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')

Phase=['Base','Anes','Both']

Part = ['13', '18', '05', '11', '22', '12', '10', '09', '19', '02', '20']
Part_nonr = ['13', '18', '05', '11', '22', '12', '10']
Part_reco=['02', '09', '19', '20']

KS=[5,6]

pdf = matplotlib.backends.backend_pdf.PdfPages("Switch_Cluster_K5_K6_wholebraind_alpha.pdf")

for p in Phase:
    if p=='Both':
        data_phase=data.query("Phase!='Reco'")
        X=data_phase.iloc[:,4:]
    else:
        data_phase=data.query("Phase=='{}'".format(p))
        X=data_phase.iloc[:,4:]

    # Assign outcome
    Y_out=np.zeros(len(X))
    Y_out[data_phase['ID'].isin(Part_reco)] = 1

    """
        K_means 7 PC
    """
    pca = PCA(n_components=7)
    pca.fit(X)
    X7 = pca.transform(X)

    for k in KS:
        kmc=KMeans(n_clusters=k, random_state=0,n_init=1000)
        kmc.fit(X7)
        P_kmc=kmc.predict(X7)

        fig, ax = plt.subplots(7, 1, figsize=(5, 20))
        fig.suptitle('Non-recovered_{}; {}_Clusters\nwholeBrain_alpha'.format(p, k), size=16)

        for t in range(0,len(Part_nonr)):
            part=Part_nonr[t]

            part_cluster = P_kmc[data_phase['ID'] == part]

            stay = 0
            switch = 0

            for l in range(1, len(part_cluster) - 1):
                if part_cluster[l] == part_cluster[l - 1]:
                    stay += 1
                if part_cluster[l] != part_cluster[l - 1]:
                    switch += 1

            ax[t].pie([stay, switch], labels=["stay", "switch"])
            ax[t].set_title('Participant: '+part)

        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(4, 1, figsize=(5, 15))
        fig.suptitle('Recovered_{}; {}_Clusters\nwholeBrain_alpha'.format(p, k), size=16)

        for t in range(0,len(Part_reco)):
            part=Part_reco[t]
            part_cluster = P_kmc[data_phase['ID'] == part]

            stay = 0
            switch = 0

            for l in range(1, len(part_cluster) - 1):
                if part_cluster[l] == part_cluster[l - 1]:
                    stay += 1
                if part_cluster[l] != part_cluster[l - 1]:
                    switch += 1

            ax[t].pie([stay, switch], labels=["stay", "switch"])
            ax[t].set_title('Participant: ' + part)
        pdf.savefig(fig)
        plt.close()

pdf.close()


