import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
matplotlib.use('Qt5Agg')
from nilearn import plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_connectivity(X_conn):
    regions = ['LF','LC','LP','LO','LT','RF','RC','RP','RO','RT']
    conn_matrix = np.zeros((len(regions), len(regions)))
    coords = np.loadtxt('helper_functions/coordinates.txt')

    for t in range(len(X_conn)):
        tmp = X_conn
        conn_tmp = pd.DataFrame(np.zeros((len(regions), len(regions))))
        conn_tmp.columns = regions
        conn_tmp.index = regions

        for i in regions:
            for a in regions:
                try:
                    conn_tmp.loc[i, a] = tmp[i + '_' + a]
                except:
                    conn_tmp.loc[i, a] = tmp[a + '_' + i]

    conn_matrix = np.array(conn_tmp)

    colormap = matplotlib.cm.get_cmap('OrRd')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.3)

    fig=plotting.plot_connectome(conn_matrix, node_coords=coords, edge_vmin=0, edge_vmax=0.3,
                             edge_cmap=colormap, colorbar=True, edge_threshold=None,
                             node_color=colormap(norm(conn_matrix.diagonal())),
                             display_mode='lzr')
    return fig

def plot_pca_results(pdf,X3,Y_out):
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    n = np.where(Y_out == 1)
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], color='red', label="Recovered Patients")
    n = np.where(Y_out == 0)
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], color='blue', label="Non-Recovered Patients")
    n = np.where(Y_out == 3)
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], color='green', label="Healthy controls")
    plt.title('{}_PCA_allPart_wholeBrain_alpha')
    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('PCA_allPart_wholeBrain_alpha', size=16)

    ax[0].set_title('PC 0 and 1')
    n = np.where(Y_out == 1)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='red', label="Recovered Patients")
    n = np.where(Y_out == 0)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='blue', label="Non-Recovered Patients")
    n = np.where(Y_out == 3)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='green', label="Healthy controls")

    ax[1].set_title('PC 1 and 2')
    n = np.where(Y_out == 1)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='red', label="Recovered Patients")
    n = np.where(Y_out == 0)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='blue', label="Non-Recovered Patients")
    n = np.where(Y_out == 3)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='green', label="Healthy controls")

    ax[2].set_title('PC 0 and 2')
    n = np.where(Y_out == 1)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='red', label="Recovered Patients")
    n = np.where(Y_out == 0)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='blue', label="Non-Recovered Patients")
    n = np.where(Y_out == 3)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='green', label="Healthy controls")

    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()


def plot_clustered_pca(pdf,X3,Y_out,P_kmc,k):
    # visualize in 3D
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    n = np.where(Y_out==1)[0]
    ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],marker='o',c=P_kmc[n],label="Recovered Patients")
    n= np.where(Y_out==0)[0]
    ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],marker='x',c=P_kmc[n],label="Non-Recovered Patients")
    n = np.where(Y_out == 3)[0]
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2],marker='.', c=P_kmc[n], label="Healthy controls")
    plt.title('{}_Clusters_allPart_wholeBrain_alpha'.format(str(k)))
    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('{}_Clusters_allPart_wholeBrain_alpha'.format(str(k)), size=16)

    ax[0].set_title('PC 0 and 1')
    n = np.where(Y_out == 1)[0]
    ax[0].scatter(X3[n, 0], X3[n, 1], marker='o', c=P_kmc[n], label="Recovered Patients")
    n = np.where(Y_out == 0)[0]
    ax[0].scatter(X3[n, 0], X3[n, 1], marker='x', c=P_kmc[n], label="Non-Recovered Patients")
    n = np.where(Y_out == 3)[0]
    ax[0].scatter(X3[n, 0], X3[n, 1], marker='.', c=P_kmc[n], label="Healthy adults")


    ax[1].set_title('PC 1 and 2')
    n = np.where(Y_out == 1)[0]
    ax[1].scatter(X3[n, 1], X3[n, 2], marker='o', c=P_kmc[n], label="Recovered Patients")
    n = np.where(Y_out == 0)[0]
    ax[1].scatter(X3[n, 1], X3[n, 2], marker='x', c=P_kmc[n], label="Non-Recovered Patients")
    n = np.where(Y_out == 3)[0]
    ax[1].scatter(X3[n, 1], X3[n, 2], marker='.', c=P_kmc[n], label="Healthy adults")

    ax[2].set_title('PC 0 and 2')
    n = np.where(Y_out == 1)[0]
    ax[2].scatter(X3[n, 0], X3[n, 2], marker='o', c=P_kmc[n], label="Recovered Patients")
    n = np.where(Y_out == 0)[0]
    ax[2].scatter(X3[n, 0], X3[n, 2], marker='x', c=P_kmc[n], label="Non-Recovered Patients")
    n = np.where(Y_out == 3)[0]
    ax[2].scatter(X3[n, 0], X3[n, 2], marker='.', c=P_kmc[n], label="Healthy adults")

    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()




def plot_explained_variance(pdf,pca):
    # PLot explained Variance
    fig = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Explained_Variance_allPart_wholeBrain_alpha')
    pdf.savefig(fig)
    plt.close()


def plot_pie_and_distribution(pdf,part,part_cluster,k):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Part {}; {}_Clusters_wholeBrain_alpha'.format(part, k), size=16)

    ax[0].plot(part_cluster)
    ax[0].set_ylim(0, k - 1)
    ax[0].set_title('Part {}; {}_Clusters_wholeBrain_alpha'.format(part, k))
    ax[0].set_ylabel('cluaster_Number')
    ax[0].set_xlabel('time')

    piedata = []
    clusternames = []
    for i in range(k):
        piedata.append(list(part_cluster).count(i))
        clusternames.append('cluster ' + str(i))

    ax[1].pie(piedata, labels=clusternames, autopct='%1.1f%%', startangle=90)
    pdf.savefig(fig)
    plt.close()
