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

#part = ['13', '18', '05', '11', '19', '02', '20', '22', '12', '10', '09']
part = ['09']

for p in part:
    pdf = matplotlib.backends.backend_pdf.PdfPages("wholebrain_Part_{}.pdf".format(p))

    connectivity=['wPLI','dPLI']

    for c in connectivity:
        data = pd.read_pickle('../data/WholeBrain_{}_10_1_alpha.pickle'.format(c))
        areas=data.columns[4:]

        data_p=data.query("ID == '{}'".format(p))
        data_p_Base=data_p.query("Phase == 'Base'")
        data_p_Anes=data_p.query("Phase == 'Anes'")
        data_p_Reco=data_p.query("Phase == 'Reco'")

        ######################
        # plot raw connectivity data
        ######################

        fig, ax = plt.subplots(3, 1, sharey=True, figsize=(9,6))
        fig.suptitle('Part: {} Raw connectivity data: {}'.format(p,c),size=16)
        plt.xlabel("time")
        ax[0].set_title('Baseline')
        ax[0].imshow(np.transpose(data_p_Base[areas]),vmin=0, vmax=0.7,cmap='jet')
        ax[0].xaxis.set_visible(False)

        ax[1].set_title('Anesthesia')
        ax[1].imshow(np.transpose(data_p_Anes[areas]),vmin=0, vmax=0.7,cmap='jet')
        ax[1].xaxis.set_visible(False)
        ax[1].set_ylabel('areas')

        ax[2].set_title('Recovery')
        im=ax[2].imshow(np.transpose(data_p_Reco[areas]),vmin=0, vmax=0.7,cmap='jet')

        fig.colorbar(im, ax=ax.ravel().tolist())
        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} Raw-plot'.format(p, c))

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
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='distortion')
        visualizer.fit(data_p_Anes[areas])        # Fit the data to the visualizer
        visualizer.show()

        # plot     silhouette: mean ratio of intra-cluster and nearest-cluster distance
        ax2 = plt.subplot(1, 3, 2)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='silhouette')
        visualizer.fit(data_p_Anes[areas])        # Fit the data to the visualizer
        visualizer.show()

        #plot      calinski_harabasz: ratio of within to between cluster dispersion
        ax3 = plt.subplot(1, 3, 3)
        model = KMeans(n_init=1000)
        visualizer = KElbowVisualizer(model, k=(2,12),timings=False,metric='calinski_harabasz')
        visualizer.fit(np.array(data_p_Anes[areas]))        # Fit the data to the visualizer
        visualizer.show()

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} K-Means Anesthesia'.format(p, c))

        ######################
        # SELF ORGANIZED MAPS
        ######################

        # 1 Baseline
        map_dim = 30
        som = MiniSom(map_dim, map_dim, len(areas), sigma=1.0, random_seed=1)
        som.train_batch(np.array(data_p_Base[areas]), num_iteration=len(data_p_Base[areas]) * 500, verbose=True)

        fig=plt.figure(figsize=(14, 15))
        plt.title('Part: {} SOM Baseline data: {}'.format(p,c),size=16)

        plt.pcolor(som.distance_map().T, cmap='bone')
        for cnt, xx in enumerate(np.array(data_p_Base[areas])):
            w = som.winner(xx)  # getting the winner
            plt.plot(w[0] + .5, w[1] + .5, 'o', markerfacecolor='None',
                     markeredgecolor='blue', markersize=12, markeredgewidth=2)
        plt.colorbar()
        plt.axis([0, map_dim, 0, map_dim])

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} SOM Base'.format(p, c))

        # 2 Anesthesia
        som = MiniSom(map_dim, map_dim, len(areas), sigma=1.0, random_seed=1)
        som.train_batch(np.array(data_p_Anes[areas]), num_iteration=len(data_p_Anes[areas]) * 500, verbose=True)

        fig=plt.figure(figsize=(14, 15))
        plt.title('Part: {} SOM Anesthesia data: {}'.format(p,c),size=16)

        plt.pcolor(som.distance_map().T, cmap='bone')
        for cnt, xx in enumerate(np.array(data_p_Anes[areas])):
            w = som.winner(xx)  # getting the winner
            plt.plot(w[0] + .5, w[1] + .5, 'o', markerfacecolor='None',
                     markeredgecolor='orange', markersize=12, markeredgewidth=2)
        plt.colorbar()
        plt.axis([0, map_dim, 0, map_dim])

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} SOM Anes'.format(p, c))


        # 3  Baseline and Anesthesia
        som = MiniSom(map_dim, map_dim, len(areas), sigma=1.0, random_seed=1)
        tmp=data_p.query("Phase != 'Reco'")
        som.train_batch(np.array(tmp[areas]), num_iteration=len(tmp[areas]) * 500, verbose=True)

        fig = plt.figure(figsize=(14, 15))
        plt.title('Part: {} SOM Baseline & Anesthesia data: {}'.format(p,c),size=16)

        t = np.zeros(len(tmp), dtype=int)
        t[tmp['Phase'] == 'Base'] = 0
        t[tmp['Phase'] == 'Anes'] = 1

        # use different colors and markers for each label
        markers = ['o', 's']
        colors = ['blue', 'orange']

        plt.pcolor(som.distance_map().T, cmap='bone')
        for cnt, xx in enumerate(np.array(tmp[areas])):
            w = som.winner(xx)  # getting the winner
            plt.plot(w[0] + .5, w[1] + .5, markers[t[cnt]], markeredgecolor=colors[t[cnt]],
                     markerfacecolor='None', markersize=12, markeredgewidth=2)
        plt.colorbar()
        plt.axis([0, map_dim, 0, map_dim])

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} SOM Both'.format(p, c))


        # Plot it in 3D *.*
        xx, yy = np.meshgrid(np.linspace(0, map_dim, map_dim), np.linspace(0, map_dim, map_dim))
        X1 = xx
        Y = yy
        Z = som.distance_map().T
        data = som.distance_map().T

        # create the figure
        fig = plt.figure(figsize=(14,15))
        fig.suptitle('Part: {} SOM 3D Baseline and Anesthesia data: {}'.format(p,c),size=16)

        # show the reference image
        ax1 = fig.add_subplot(121)
        ax1.imshow(data, cmap='bone', interpolation='nearest', origin='lower')

        # show the 3D rotated projection
        ax2 = fig.add_subplot(122, projection='3d')
        cset = ax2.contourf(X1, Y, Z, 1000,cmap='bone')
        plt.colorbar(cset)

        pdf.savefig(fig)
        plt.close()
        print('finished Part {}: data: {} SOM '.format(p, c))


    pdf.close()
    print('######################################## finished Part {}: '.format(p))





