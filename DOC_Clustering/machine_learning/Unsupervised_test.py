import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import sys
import os
sys.path.append('../')

data = pd.read_pickle('data/WholeBrain_wPLI_10_1_alpha.pickle')
X= data.iloc[:,4:]

Part_chro=['13','22','10', '18','05','12','11']
Part_reco=['19','20','02','09']


data_Base=data.query("Phase=='Base'")
X_Base= data_Base.iloc[:,4:]
data_Anes=data.query("Phase=='Anes'")
X_Anes= data_Anes.iloc[:,4:]
data_Reco=data.query("Phase=='Reco'")
X_Reco= data_Reco.iloc[:,4:]

pca = PCA(n_components=3)
pca.fit(X_Anes)
X = pca.transform(X_Anes)


#Plot the data in 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0],
           X[:, 1],
           X[:, 2],edgecolors='Blue')


#Generate some data in 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[np.where(data['Phase']=='Base')[0], 0],
           X[np.where(data['Phase']=='Base')[0], 1],
           X[np.where(data['Phase']=='Base')[0], 2],edgecolors='Blue')
ax.scatter(X[np.where(data['Phase']=='Anes')[0], 0],
           X[np.where(data['Phase']=='Anes')[0], 1],
           X[np.where(data['Phase']=='Anes')[0], 2],edgecolors='Red')
ax.scatter(X[np.where(data['Phase']=='Reco')[0], 0],
           X[np.where(data['Phase']=='Reco')[0], 1],
           X[np.where(data['Phase']=='Reco')[0], 2],edgecolors='Yellow')
plt.show()

# why do we fit only on x and not y or both?

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('Elbow Method xy')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

