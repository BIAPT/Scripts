import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import nilearn
matplotlib.use('Qt5Agg')
from nilearn import plotting
import time
from tqdm import tqdm

Phase = 'Base'
Part = 'S09'

data = pd.read_pickle('data/New_Part_WholeBrain_wPLI_10_1_alpha.pickle')
X=data.query("Phase=='{}'".format(Phase))
X=X.query("ID=='{}'".format(Part))
X_conn=X.iloc[:,4:]

regions = ['LF','LC','LP','LO','LT','RF','RC','RP','RO','RT']
conn_matrix = np.zeros(( len(X_conn),len(regions), len(regions)))

coords = np.loadtxt('visualization/coordinates.txt')

for t in range(len(X_conn)):
    tmp = X_conn.iloc[t,:]
    conn_tmp = pd.DataFrame(np.zeros((len(regions), len(regions))))
    conn_tmp.columns = regions
    conn_tmp.index = regions

    for i in regions:
        for a in regions:
            if a==i=='RC':
                # ERROR IN DF NAMES: RC_RC DOES NOT EXIST
                tmp[i + '_' + a] = 0
            try:
                conn_tmp.loc[i, a] = tmp[i + '_' + a]
            except:
                conn_tmp.loc[i, a] = tmp[a + '_' + i]

    conn_matrix[t] = conn_tmp

#plotting.plot_matrix(conn_tmp)
#plotting.show()

colormap = matplotlib.cm.get_cmap('OrRd')
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)

for t, conn_mat_t in enumerate(tqdm(conn_matrix)):
    plotting.plot_connectome(conn_mat_t, node_coords=coords, edge_vmin=0, edge_vmax=0.5,
                             edge_cmap=colormap, colorbar=True, edge_threshold=None,
                             node_color=colormap(norm(conn_mat_t.diagonal())),
                             display_mode='lzr')
    plt.suptitle('WSAS {}_{} '.format(Part,Phase))
    plt.savefig('video_images/'+str(t)+".png")
    plt.close()

#node_color= (colormap(conn_mat_t.diagonal()))
import cv2
import os

image_folder = 'video_images'
video_name = '{}_{}.wmv'.format(Part, Phase)

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 5, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

#cv2.destroyAllWindows()
video.release()
