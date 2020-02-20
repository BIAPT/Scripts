import scipy.io
import extract_features
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import stability_measure
from matplotlib import pyplot as plt


mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
X_anes_step=extract_features.extract_features(data_anes_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
X_rest_step=extract_features.extract_features(data_rest_step)

X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)

"""
Stability Index
"""
P=[2,3,4,5]     #number of Principal components to iterate
K=[2,3,4,5,6]     #number of K-clusters to iterate
Rep=2          #number of Repetitions (Mean at the end)

X_temp=X_rest_step    #Template set (50% of Participants)
X_test=X_anes_step    #Test set (50% of Participants)

[SI_M ,SI_SD] = stability_measure.compute_stability_index(X_temp, X_test, P, K, Rep)


P_l="".join(map(str, P))
K_l="".join(map(str, K))

plt.imshow(SI_M)
plt.title('Stability Index Mean')
plt.colorbar()
plt.xlabel('Principle Components')
plt.xticks(np.arange(len(P)),P_l)
plt.ylabel('K-Clusters')
plt.yticks(np.arange(len(K)),K_l)

plt.imshow(SI_SD)
plt.colorbar()
plt.title('Stability Index SD')
plt.xlabel('Principle Components')
plt.xticks(np.arange(len(P)),P_l)
plt.ylabel('K-Clusters')
plt.yticks(np.arange(len(K)),K_l)


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
