import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
matplotlib.use('Qt5Agg')
from nilearn import plotting

def plot_connectivity(X_conn):
    regions = ['LF','LC','LP','LO','LT','RF','RC','RP','RO','RT']
    conn_matrix = np.zeros((len(regions), len(regions)))
    coords = np.loadtxt('visualization/coordinates.txt')

    for t in range(len(X_conn)):
        tmp = X_conn
        conn_tmp = pd.DataFrame(np.zeros((len(regions), len(regions))))
        conn_tmp.columns = regions
        conn_tmp.index = regions

        for i in regions:
            for a in regions:
                # only because of error
                if i == 'RC' and a == 'RC':
                    conn_tmp.loc[i, a] = tmp['LC' + '_' + 'LC']
                else:
                    try:
                        conn_tmp.loc[i, a] = tmp[i + '_' + a]
                    except:
                        conn_tmp.loc[i, a] = tmp[a + '_' + i]

    conn_matrix = np.array(conn_tmp)

    #plotting.plot_matrix(conn_matrix)
    #plotting.show()

    colormap = matplotlib.cm.get_cmap('OrRd')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.3)

    fig=plotting.plot_connectome(conn_matrix, node_coords=coords, edge_vmin=0, edge_vmax=0.3,
                             edge_cmap=colormap, colorbar=True, edge_threshold=None,
                             node_color=colormap(norm(conn_matrix.diagonal())),
                             display_mode='lzr')
    return fig
