import scipy.io
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import stability_measure
from matplotlib import pyplot as plt
import random
import pickle

data=pd.read_pickle('data/final_wPLI_clustering.pickle')
X=data.iloc[:,4:]
Y_ID=data.iloc[:,1]
Y_St=data.iloc[:,2]
Y_time=data.iloc[:,3]



"""
Stability Index
"""
P=[3,4,5,6,7,8,9,10]     #number of Principal components to iterate
K=[2,3,4,5,6,7,8,9,10]     #number of K-clusters to iterate
Rep=100         #number of Repetitions (Mean at the end)

[SI_M ,SI_SD] = stability_measure.compute_stability_index(X,Y_ID, P, K, Rep)


pd.DataFrame(SI_M).to_pickle('data/SI_M_100x100x3-10x2-10.pickle')
pd.DataFrame(SI_SD).to_pickle('data/SI_SD_100x100x3-10x2-10.pickle')



P_l="".join(map(str, P))
K_l="".join(map(str, K))

plt.imshow(np.transpose(SI_M))
plt.xticks(np.arange(10),('2','3','4','5','6','7','8','9','10'))
plt.yticks(np.arange(9),('3','4','5','6','7','8','9','10'))
plt.imshow(np.transpose(SI_M))
plt.title('Stability Index Mean')
plt.colorbar()
plt.ylabel('Principle Components')
plt.xlabel('K-Clusters')
#plt.yticks(np.arange(len(K)),K_l)


plt.imshow(np.transpose(SI_M[3:,:]))
plt.xticks(np.arange(7),('5','6','7','8','9','10'))
plt.yticks(np.arange(9),('3','4','5','6','7','8','9','10'))
plt.imshow(np.transpose(SI_M))
plt.title('Stability Index Mean')
plt.colorbar()
plt.clim(0.5,0.9)
plt.ylabel('Principle Components')
plt.xlabel('K-Clusters')
#plt.yticks(np.arange(len(K)),K_l)

plt.imshow(SI_M)
plt.yticks(np.arange(10),('2','3','4','5','6','7','8','9','10'))
plt.xticks(np.arange(9),('3','4','5','6','7','8','9','10'))
plt.imshow(SI_M)
plt.title('Stability Index Mean')
plt.colorbar()
plt.xlabel('Principle Components')
plt.ylabel('K-Clusters')
#plt.yticks(np.arange(len(K)),K_l)

plt.imshow(np.transpose(SI_SD))
plt.xticks(np.arange(10),('2','3','4','5','6','7','8','9','10'))
plt.yticks(np.arange(9),('3','4','5','6','7','8','9','10'))
plt.imshow(np.transpose(SI_SD))
plt.title('Stability Index SD')
plt.colorbar()
plt.ylabel('Principle Components')
plt.xlabel('K-Clusters')

"""
Silhouette Score
"""
P=[2,3,4,5]     #number of Principal components to iterate
K=[2,3,4,5,6]     #number of K-clusters to iterate
Rep=2          #number of Repetitions (Mean at the end)

X_temp=X_rest_step    #Template set (50% of Participants)
X_test=X_anes_step    #Test set (50% of Participants)

[SIL_M ,SIL_SD] = stability_measure.compute_silhouette_score(X_temp, X_test, P, K, Rep)




P_l="".join(map(str, P))
K_l="".join(map(str, K))

plt.imshow(SIL_M)
plt.title('Silhouette Score Mean')
plt.colorbar()
plt.xlabel('Principle Components')
plt.xticks(np.arange(len(P)),P_l)
plt.ylabel('K-Clusters')
plt.yticks(np.arange(len(K)),K_l)

plt.imshow(SIL_SD)
plt.title('Silhouette Score SD')
plt.colorbar()
plt.xlabel('Principle Components')
plt.xticks(np.arange(len(P)),P_l)
plt.ylabel('K-Clusters')
plt.yticks(np.arange(len(K)),K_l)
