import scipy
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from prepareDataset import *
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

X=X.iloc[:,empty==0]
X_Anes=X_Anes.iloc[:,empty==0]
X_Base=X_Base.iloc[:,empty==0]
X_Reco=X_Reco.iloc[:,empty==0]

Y_out_int_Base=np.zeros(len(Y_out_Base))
Y_out_int_Base[Y_out_Base=='1']=1

Y_out_int_Anes=np.zeros(len(Y_out_Anes))
Y_out_int_Anes[Y_out_Anes=='1']=1

Y_out_int_Reco=np.zeros(len(Y_out_Reco))
Y_out_int_Reco[Y_out_Reco=='1']=1


"""
    PCA - 3D visualization
"""

# Baseline

kmc=KMeans(n_clusters=2, random_state=0,n_init=1000)
kmc.fit(X_Base)
P_kmc2=kmc.predict(X_Base)
accuracy_score(Y_out_int_Base, P_kmc2)

pca = PCA(n_components=3)

pca.fit(X_Base)
X3 = pca.transform(X_Base)

fig = plt.figure()
ax = Axes3D(fig)
n=np.where([(Y_out_Base=='1') & (P_kmc2==1)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='v',color='red',label="Recovered_correct classified")
n=np.where([(Y_out_Base=='1') & (P_kmc2==0)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Recovered_misclassified")
n=np.where([(Y_out_Base=='0') & (P_kmc2==0)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='v',color='blue',label="Chronic_correct classified")
n=np.where([(Y_out_Base=='0') & (P_kmc2==1)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="Chronic_misclassified")
plt.title('Baseline')
plt.legend()


# Anesthesia
kmc=KMeans(n_clusters=2, random_state=0, n_init=1000)
kmc.fit(X_Anes)
P_kmc2=kmc.predict(X_Anes)
accuracy_score(Y_out_int_Anes, P_kmc2)

pca = PCA(n_components=3)
pca.fit(X_Anes)
X3 = pca.transform(X_Anes)

fig = plt.figure()
ax = Axes3D(fig)
n=np.where([(Y_out_Anes=='1') & (P_kmc2==1)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='v',color='red',label="Recovered_correct classified")
n=np.where([(Y_out_Anes=='1') & (P_kmc2==0)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Recovered_misclassified")
n=np.where([(Y_out_Anes=='0') & (P_kmc2==0)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='v',color='blue',label="Chronic_correct classified")
n=np.where([(Y_out_Anes=='0') & (P_kmc2==1)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="Chronic_misclassified")
plt.title('Anesthesia')
plt.legend()


# Recovery
kmc=KMeans(n_clusters=2, random_state=0,n_init=1000)
kmc.fit(X_Reco)
P_kmc2=kmc.predict(X_Reco)
accuracy_score(Y_out_int_Reco, P_kmc2)

pca = PCA(n_components=3)
pca.fit(X_Reco)
X3 = pca.transform(X_Reco)

fig = plt.figure()
ax = Axes3D(fig)
n=np.where([(Y_out_Reco=='1') & (P_kmc2==1)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='v',color='red',label="Recovered_correct classified")
n=np.where([(Y_out_Reco=='1') & (P_kmc2==0)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Recovered_misclassified")
n=np.where([(Y_out_Reco=='0') & (P_kmc2==0)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='v',color='blue',label="Chronic_correct classified")
n=np.where([(Y_out_Reco=='0') & (P_kmc2==1)])
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="Chronic_misclassified")
plt.title('Recovery')
plt.legend()


# PLot explained Variance
pca = PCA()
pca.fit(X_Base)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xlim(1,20)

pca = PCA()
pca.fit(X_Anes)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

pca = PCA()
pca.fit(X_Reco)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.title('Explained Variance PCA')
plt.legend(['Baseline','Anesthesia','Recovery'])



