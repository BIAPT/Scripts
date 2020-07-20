import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from minisom import MiniSom
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys
sys.path.append('../')
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.backends.backend_pdf

part = ['WSAS13', 'WSAS18', 'WSAS05', 'WSAS11', 'WSAS19', 'WSAS02', 'WSAS20', 'WSAS22', 'WSAS12', 'WSAS10', 'WSAS09']

pdf = matplotlib.backends.backend_pdf.PdfPages("F_C_P_Part_Summary.pdf")

for p in part:

    connectivity=['wPLI']

    for c in connectivity:
        data=pd.read_pickle('data/F_C_P_wPLI_30_10_allfrequ.pickle')
        areas=data.columns[4:]

        data_p=data.query("ID == '{}'".format(p))
        data_p_Base=data_p.query("Phase == 'Base'")
        data_p_Anes=data_p.query("Phase == 'Anes'")
        data_p_Reco=data_p.query("Phase == 'Reco'")


        ######################
        # PCA of all 3 phases:
        ######################
        pca = PCA(n_components=3)
        pca.fit(data_p[areas])
        X3_B = pca.transform(data_p_Base[areas])
        X3_A = pca.transform(data_p_Anes[areas])
        X3_R = pca.transform(data_p_Reco[areas])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X3_B[:, 0], X3_B[:, 1], X3_B[:, 2], marker='v', color='blue', label="Baseline")
        ax.scatter(X3_A[:, 0], X3_A[:, 1], X3_A[:, 2], marker='o', color='orange', label="Anesthesia")
        ax.scatter(X3_R[:, 0], X3_R[:, 1], X3_R[:, 2], marker='v', color='green', label="Recovery")
        plt.legend()
        plt.title('Part: {} PCA connectivity data: {}'.format(p,c),size=16)
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(1, 3,figsize=(12,6))
        fig.suptitle('Part: {} PCA connectivity data: {}'.format(p,c),size=16)

        ax[0].set_title('PC 0 and 1')
        ax[0].scatter(X3_B[:, 0], X3_B[:, 1], color='blue', label="Baseline")
        ax[0].scatter(X3_A[:, 0], X3_A[:, 1], color='orange', label="Anesthesia")
        ax[0].scatter(X3_R[:, 0], X3_R[:, 1], color='green', label="Recovery")

        ax[1].set_title('PC 0 and 2')
        ax[1].scatter(X3_B[:, 0], X3_B[:, 2], color='blue', label="Baseline")
        ax[1].scatter(X3_A[:, 0], X3_A[:, 2], color='orange', label="Anesthesia")
        ax[1].scatter(X3_R[:, 0], X3_R[:, 2], color='green', label="Recovery")

        ax[2].set_title('PC 1 and 2')
        ax[2].scatter(X3_B[:, 2], X3_B[:, 1], color='blue', label="Baseline")
        ax[2].scatter(X3_A[:, 2], X3_A[:, 1], color='orange', label="Anesthesia")
        ax[2].scatter(X3_R[:, 2], X3_R[:, 1], color='green', label="Recovery")

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} PCA'.format(p, c))

        ######################
        # K-Means Clustering Baseline
        ######################

        fig=plt.figure(figsize=(17,8))
        fig.suptitle('Part: {} Clustering Baseline data: {}'.format(p,c),size=16)
        plt.subplot(131)

        # plot     distortion: mean sum of squared distances to centers
        ax1 = plt.subplot(1, 3, 1)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='distortion', ax = ax1)
        visualizer.fit(data_p_Base[areas])        # Fit the data to the visualizer
        #visualizer.show()

        # plot     silhouette: mean ratio of intra-cluster and nearest-cluster distance
        ax2 = plt.subplot(1, 3, 2)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='silhouette', ax = ax2)
        visualizer.fit(data_p_Base[areas])        # Fit the data to the visualizer
        #visualizer.show()

        #plot      calinski_harabasz: ratio of within to between cluster dispersion
        ax3 = plt.subplot(1, 3, 3)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='calinski_harabasz', ax = ax3)
        visualizer.fit(np.array(data_p_Base[areas]))        # Fit the data to the visualizer
        #visualizer.show()

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} K-Means Base'.format(p,c))

        ######################
        # K-Means Clustering Anesthesia
        ######################

        fig=plt.figure(figsize=(17,8))
        fig.suptitle('Part: {} Clustering Anesthesia data: {}'.format(p,c),size=16)
        plt.subplot(131)

        # plot     distortion: mean sum of squared distances to centers
        ax1 = plt.subplot(1, 3, 1)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='distortion', ax = ax1)
        visualizer.fit(data_p_Anes[areas])        # Fit the data to the visualizer
        #visualizer.show()

        # plot     silhouette: mean ratio of intra-cluster and nearest-cluster distance
        ax2 = plt.subplot(1, 3, 2)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='silhouette', ax= ax2)
        visualizer.fit(data_p_Anes[areas])        # Fit the data to the visualizer
        #visualizer.show()

        #plot      calinski_harabasz: ratio of within to between cluster dispersion
        ax3 = plt.subplot(1, 3, 3)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='calinski_harabasz', ax = ax3)
        visualizer.fit(np.array(data_p_Anes[areas]))        # Fit the data to the visualizer
        #visualizer.show()

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} K-Means Anesthesia'.format(p, c))

pdf.close()