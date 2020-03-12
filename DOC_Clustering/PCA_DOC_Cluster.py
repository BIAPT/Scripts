import scipy
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd

data=pd.read_pickle('data/final_wPLI_clustering.pickle')
X=data.iloc[:,4:]
Y_ID=data.iloc[:,1]
Y_St=data.iloc[:,2]
Y_time=data.iloc[:,3]

"""
    PCA - 3D visualization
"""
pca = PCA(n_components=3)
pca.fit(X)
X3 = pca.transform(X)


fig = plt.figure()
ax = Axes3D(fig)
n=np.where(Y_St=='Anes')
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Anesthesia", edgecolor='k')
n=np.where(Y_St=='Base')
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="Resting State", edgecolor='k')
n=np.where(Y_St=='Reco')
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="Recovery", edgecolor='k')
ax.legend()
#plt.title('Anesthesia 2-Clustering')
#plt.savefig("k_2_cluster",bbox_inches='tight')


fig = plt.figure()
ax = Axes3D(fig)
n=np.where(Y_ID=='WSAS02')
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='+',color='red',label="WSAS02 +", edgecolor='k')
n=np.where(Y_ID=='WSAS20')
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='+',color='orange',label="WSAS20 +", edgecolor='k')
n=np.where(Y_ID=='WSAS05')
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="WSAS05 -", edgecolor='k')
n=np.where(Y_ID=='WSAS12')
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="WSAS12 -", edgecolor='k')
ax.legend()