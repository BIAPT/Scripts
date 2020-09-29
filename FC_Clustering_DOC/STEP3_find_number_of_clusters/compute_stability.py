import scipy.io
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import sys
import os
import pandas as pd
sys.path.append('../')
from helper_functions import stability_measure
from matplotlib import pyplot as plt
import random
import pickle
import matplotlib.backends.backend_pdf
from helper_functions.General_Information import *

pdf = matplotlib.backends.backend_pdf.PdfPages("SI_SIL_New_group_wholebrain_wPLI_30_10_allfrequ.pdf")

# random data
mean = np.mean(X)
std = np.std(X)

data_random= np.random.normal(mean, std, size=X.shape)
Y_ID_random = data['ID']


"""
Stability Index
"""
P=[3,4,5,6,7,8,9,10]     #number of Principal components to iterate
K=[2,3,4,5,6,7,8,9,10]     #number of K-clusters to iterate
Rep=10        #number of Repetitions (Mean at the end)

[SI_M_rand ,SI_SD_rand] = stability_measure.compute_stability_index(data_random, Y_ID_random, P, K, Rep)
[SI_M_Anes ,SI_SD_Anes] = stability_measure.compute_stability_index(X_Anes, Y_ID_Anes, P, K, Rep)
[SI_M_Base ,SI_SD_Base] = stability_measure.compute_stability_index(X_Base, Y_ID_Base, P, K, Rep)
[SI_M_BA ,SI_SD_BA] = stability_measure.compute_stability_index(X_BA, Y_ID_BA, P, K, Rep)

plt.figure()
plt.show()
fig,a =  plt.subplots(4,2)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9,10] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')

im=a[0][0].imshow(np.transpose(SI_M_rand))
a[0][0].set_title('Stability Index Mean: Random')
a[0][0].set_xlabel("")
plt.colorbar(im,ax=a[0,0])

im=a[0][1].imshow(np.transpose(SI_SD_rand))
a[0][1].set_xlabel("")
a[0][1].set_title('Stability Index SD: Random')
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[0,1])

im=a[1][0].imshow(np.transpose(SI_M_Base))
a[1][0].set_title('Stability Index Mean: Baseline')
a[1][0].set_xlabel("")
im.set_clim(0.2,0.65)
plt.colorbar(im,ax=a[1,0])

im=a[1][1].imshow(np.transpose(SI_SD_Base))
a[1][1].set_title('Stability Index SD: Baseline')
a[1][1].set_xlabel("")
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[1,1])


im=a[2][0].imshow(np.transpose(SI_M_Anes))
a[2][0].set_title('Stability Index Mean: Anesthesia')
a[2][0].set_xlabel("")
im.set_clim(0.2,0.65)
plt.colorbar(im,ax=a[2,0])

im=a[2][1].imshow(np.transpose(SI_SD_Anes))
a[2][1].set_title('Stability Index SD: Anesthesia')
a[2][1].set_xlabel("")
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[2,1])

im=a[3][0].imshow(np.transpose(SI_M_BA))
a[3][0].set_title('Stability Index Mean: Anesthesia and Baseline')
im.set_clim(0.2,0.65)
plt.colorbar(im,ax=a[3,0])


im=a[3][1].imshow(np.transpose(SI_SD_BA))
a[3][1].set_title('Stability Index SD: Anesthesia and Baseline')
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[3,1])

fig.set_figheight(17)
fig.set_figwidth(10)

pdf.savefig(fig)


pd.DataFrame(SI_M_Base).to_pickle('SI_Base_F_C_P_wPLI_30_10_allfr.pickle')
pd.DataFrame(SI_M_Anes).to_pickle('SI_Anes_F_C_P_wPLI_30_10_allfr.pickle')
pd.DataFrame(SI_M_rand).to_pickle('SI_rand_F_C_P_wPLI_30_10_allfr.pickle')
pd.DataFrame(SI_M_BA).to_pickle('SI_BA_F_C_P_wPLI_30_10_allfr.pickle')


"""
Silhouette Score
"""
P=[3,4,5,6,7,8,9,10]     #number of Principal components to iterate
K=[2,3,4,5,6,7,8,9,10]     #number of K-clusters to iterate

SIS_Rand = stability_measure.compute_silhouette_score(data_random, P, K)
SIS_Anes = stability_measure.compute_silhouette_score(X_Anes, P, K)
SIS_Base = stability_measure.compute_silhouette_score(X_Base, P, K)
SIS_BA = stability_measure.compute_silhouette_score(X_BA, P, K)

fig,a =  plt.subplots(1,4)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')
im=a[0].imshow(np.transpose(SIS_Rand),cmap='viridis_r')
a[0].set_title('Silhouette Score  : Random')
a[0].set_xlabel("")
plt.colorbar(im,ax=a[0])
im=a[1].imshow(np.transpose(SIS_Base),cmap='viridis_r')
a[1].set_title('Silhouette Score : Baseline')
a[1].set_xlabel("")
im.set_clim(0.1,0.45)
plt.colorbar(im,ax=a[1])
im=a[2].imshow(np.transpose(SIS_Anes),cmap='viridis_r')
a[2].set_title('Silhouette Score  : Anesthesia')
a[2].set_xlabel("")
im.set_clim(0.1,0.45)
plt.colorbar(im,ax=a[2])
im=a[3].imshow(np.transpose(SIS_BA),cmap='viridis_r')
a[3].set_title('Silhouette Score  : Baseline and Anesthesia')
im.set_clim(0.1,0.45)
plt.colorbar(im,ax=a[3])
fig.set_figheight(3)
fig.set_figwidth(20)
pdf.savefig(fig)

plt.show()

pdf.close()


pd.DataFrame(SIS_Base).to_pickle('SIS_Base_F_C_P_wPLI_30_10_allfr.pickle')
pd.DataFrame(SIS_Anes).to_pickle('SIS_Anes_F_C_P_wPLI_30_10_allfr.pickle')
pd.DataFrame(SIS_Rand).to_pickle('SIS_rand_F_C_P_wPLI_30_10_allfr.pickle')
pd.DataFrame(SIS_BA).to_pickle('SIS_BA_F_C_P_wPLI_30_10_allfr.pickle')



# Try k=3, p=10 on Anesthesia set

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
p=10
k=5

pca = PCA(n_components=p)
pca.fit(X_Anes)
X_LD = pca.transform(X_Anes) # get a low dimension version of X_temp

kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
kmeans.fit(X_LD)           #fit the classifier on X_template
S_pred = kmeans.predict(X_LD)

plt.plot(S_pred)