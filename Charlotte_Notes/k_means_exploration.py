import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from load_data import *
from sklearn.decomposition import PCA


#Generate some data in 3d
fig = plt.figure()
ax = Axes3D(fig)


wcss_rest = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_rest_step)
    wcss_rest.append(kmeans.inertia_)
    print(str(i))

wcss_anes = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_anes_step)
    wcss_anes.append(kmeans.inertia_)
    print (str(i))

# copy X_anesthesia step and X_rest_step together
# generate Y with 1= anesthesia and 0= resting state
wcss_all = []
#X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)
#Y_all=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0])),axis=0)
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(np.transpose(X_all))
    wcss_all.append(kmeans.inertia_)
    print (str(i))

plt.figure()
plt.plot(range(1, 20), wcss_rest,color="red")
plt.plot(range(1, 20), wcss_anes,color="blue")
plt.plot(range(1, 20), wcss_all,color="orange")
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend(["resting state","anesthesia","both"])
plt.show()
plt.savefig("ellbow_rest_anes_all.png",bbox_inches='tight')

#from yellowbrick.cluster import KElbowVisualizer, doesn't work that well


"""
    FINAL CLUSTERING
"""
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=100, random_state=0)
kmeans.fit(X_all)
Z_all = kmeans.predict(X_all)

pca3 = PCA(n_components=3)
pca3.fit(X_all)
X3 = pca3.transform(X_all)

""" Normal Plot 3D"""
fig = plt.figure()
ax = Axes3D(fig)
n = np.where(Y_all == 1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Anesthesia", edgecolor='k')
n = np.where(Y_all == 0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="Resting_state", edgecolor='k')

n = np.where(Z_all == 2)
ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], marker='o',color='lightblue',label="C3", edgecolor='k')
n = np.where(Z_all == 3)
ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], marker='o',color='yellow',label="C4", edgecolor='k')
ax.legend()


pca3 = PCA(n_components=3)
pca3.fit(X_anes_step)
X3 = pca3.transform(X_anes_step)

""" Normal Plot 3D"""
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X3[:, 0], X3[:, 1],X3[:, 2], marker='o',color='red',label="Anesthesia", edgecolor='k')


pca3 = PCA(n_components=3)
pca3.fit(X_rest_step)
X3 = pca3.transform(X_rest_step)

""" Normal Plot 3D"""
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X3[:, 0], X3[:, 1],X3[:, 2], marker='o',color='blue',label="Rest", edgecolor='k')



"""
#######################################
        PLOT FOR DIFFERENCE MATRIX
#######################################
"""

wcss_rest = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_rest_step_diff)
    wcss_rest.append(kmeans.inertia_)
    print(str(i))

wcss_anes = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_anes_step_diff)
    wcss_anes.append(kmeans.inertia_)
    print (str(i))

# copy X_anesthesia step and X_rest_step together
# generate Y with 1= anesthesia and 0= resting state
wcss_all = []
time_anes=np.array(range(1,len(X_anes_step)+1))
time_rest=np.array(range(1,len(X_rest_step)+1))


for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_all)
    wcss_all.append(kmeans.inertia_)
    print (str(i))

plt.plot(range(1, 20), wcss_rest,color="red")
plt.plot(range(1, 20), wcss_anes,color="blue")
plt.plot(range(1, 7), wcss_all,color="orange")
plt.title('Elbow Method_contrast 0.1s')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend(["resting state","anesthesia","both"])
plt.show()
plt.savefig("contrast_ellbow_rest_anes_all.png",bbox_inches='tight')



"""
#######################################
    RUN PCA 2D
#######################################
"""

pca = PCA(n_components=2)
pca.fit(X_all)
X = pca.transform(X_all)
X.shape
""" Normal Plot 2D"""
fig, ax = plt.subplots()
n = np.where(Y_all == 1)
ax.scatter(X[n, 0], X[n, 1], marker='o',color='red',label="Anesthesia", edgecolor='k')
n = np.where(Y_all == 0)
ax.scatter(X[n, 0], X[n, 1], marker='o',color='blue',label="Resting State", edgecolor='k')


""" TIME Dimension Plot 2D"""
fig, ax = plt.subplots()
X_A = X[np.where(Y_all == 1)]
for i in range(0,len(X_A)):
    ax.scatter(X_A[i, 0], X_A[i, 1], marker='o',alpha=(i/len(X_A)),color='red', label="Anesthesia", edgecolor='k')
X_R = X[np.where(Y_all == 0)]
for i in range(0,len(X_R)):
    ax.scatter(X_R[i, 0], X_R[i, 1], marker='o',alpha=(i/len(X_R)),color='blue', label="Resting State", edgecolor='k')


"""
#######################################
    RUN PCA 3D
#######################################
"""

pca3 = PCA(n_components=3)
pca3.fit((X_all))
X3 = pca3.transform((X_all))

""" Normal Plot 3D"""
fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = np.where(Y_all == 1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Anesthesia", edgecolor='k')
n = np.where(Y_all == 0)
ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], marker='o',color='blue',label="Resting State", edgecolor='k')
ax.legend()

""" TIME Dimension Plot 3D"""
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
X_A = X3[np.where(Y_all == 1)]
for i in range(0,len(X_A)):
    ax.scatter(X_A[i, 0], X_A[i, 1], X_A[i, 2], marker='o',alpha=(i/len(X_A)),color='red', label="Anesthesia", edgecolor='k')
X_R = X3[np.where(Y_all == 0)]
for i in range(0,len(X_R)):
    ax.scatter(X_R[i, 0], X_R[i, 1],X_R[i, 2], marker='o',alpha=(i/len(X_R)),color='blue', label="Resting State", edgecolor='k')


""" Contrast Plot 3D"""
X_all_diff= np.concatenate((X_anes_step_diff,X_rest_step_diff),axis=0)
Y_all_diff=np.concatenate((np.ones(X_anes_step_diff.shape[0]),np.zeros(X_rest_step_diff.shape[0])),axis=0)

pca3 = PCA(n_components=3)
pca3.fit(X_all_diff)
X3_diff = pca3.transform(X_all_diff)

fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_A_diff = X3_diff[np.where(Y_all_diff == 1)]
for i in range(0,len(X_A_diff)):
    ax.scatter(X_A_diff[i, 0], X_A_diff[i, 1], X_A_diff[i, 2], marker='o',alpha=(i/len(X_A_diff)),color='red', label="Anesthesia", edgecolor='k')
X_R_diff = X3_diff[np.where(Y_all_diff == 0)]
for i in range(0,len(X_R_diff)):
    ax.scatter(X_R_diff[i, 0], X_R_diff[i, 1],X_R_diff[i, 2], marker='o',alpha=(i/len(X_R_diff)),color='blue', label="Resting State", edgecolor='k')


fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = np.where(Y_all_diff == 1)
ax.scatter(X3_diff[n, 0], X3_diff[n, 1],X3_diff[n, 2], marker='o',color='red',label="Anesthesia", edgecolor='k')
n = np.where(Y_all_diff == 0)
ax.scatter(X3_diff[n, 0], X3_diff[n, 1], X3_diff[n, 2], marker='o',color='blue',label="Resting State", edgecolor='k')
ax.legend()

fig= plt.figure()
plt.imshow(X_anes_step,cmap="jet")
fig= plt.figure()
plt.imshow(data_anes_avg,cmap="jet")

fig= plt.figure()
plt.imshow(X_rest_step,cmap="jet")
fig= plt.figure()
plt.imshow(data_rest_avg,cmap="jet")


"""
#######################################
    PCA INTERPRETATION

https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

we could also use PCA to de noise the data

#######################################
"""

pca = PCA().fit(X_all)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

