import scipy.io
import extract_features
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_avg.mat')
data_anes_avg = mat['result_wpli_anes_avg']
X_rest_avg=extract_features.extract_features(data_anes_avg)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_avg.mat')
data_rest_avg = mat['result_wpli_rest_avg']
X_rest_avg=extract_features.extract_features(data_rest_avg)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
X_anes_step=extract_features.extract_features(data_anes_step)
#X_anes_step_diff = extract_features.get_difference(X_anes_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes5_step.mat')
data_anes5_step = mat['result_wpli_anes5_step']
X_anes5_step=extract_features.extract_features(data_anes5_step)
#X_anes5_step_diff = extract_features.get_difference(X_anes5_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
X_rest_step=extract_features.extract_features(data_rest_step)
#X_rest_step_diff = extract_features.get_difference(X_rest_step)

X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)
#X_all= np.concatenate((X_anes_step[:10,:],X_rest_step[:10,:]),axis=0)
Y_all=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0])),axis=0)
#Y_all=np.concatenate((np.ones(10),np.zeros(10)),axis=0)

X_all_2= np.concatenate((X_anes_step,X_rest_step,X_anes5_step),axis=0)
Y_all_2=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0]), np.repeat(2, X_anes5_step.shape[0])),axis=0)

"""
    Compute Stability index
"""
P=[2,3,4,5]     #number of Principal components to iterate
K=[2,3,4,5]     #number of K-clusters to iterate
Rep=10          #number of Repetitions (Mean at the end)

X_temp=X_all    #Template set (50% of Participants)
X_test=X_all    #Test set (50% of Participants)
X_complete = np.row_stack([X_temp,X_test]) #complete input set for PCA-fit

SI_M=np.zeros([len(K) ,len(P)])     # Mean stability index
SI_SD=np.zeros([len(K) ,len(P)])    # stability index SD
SI=np.zeros([Rep,len(K) ,len(P)])   # Collection of stability index over Repetitions
from scipy.spatial import distance
from tqdm import tqdm

for r in tqdm(range(0,Rep)):
    p_i = 0
    for p in P:
        pca3 = PCA(n_components=p)
        pca3.fit(X_complete)
        X_temp_LD = pca3.transform(X_temp) # get a low dimension version of X_temp
        X_test_LD = pca3.transform(X_test) # and X_test

        k_i=0
        for k in K:
            kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=1)
            kmeans.fit(X_temp_LD)           #fit the classifier on X_template
            S_temp = kmeans.predict(X_test_LD)

            kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=1)
            kmeans.fit(X_test_LD)           #fit the classifier on X_test
            S_test = kmeans.predict(X_test_LD)

            # proportion of disagreeing components in u and v
            SI[r,p_i,k_i]=distance.hamming(S_test,S_temp) # should be already normalized
            k_i=k_i+1

        # increase p iteration by one
        p_i=p_i+1
        print('PC '+str(p)+' finished' )

SI_M=np.mean(SI,axis=0)
SI_D=np.std(SI,axis=0)


"""
    Cluster ANALYSIS- 3d visualization
"""
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=1000, random_state=0)
kmeans.fit(X_all)
Z_step_7 = kmeans.predict(X_all)
Z_anes5 = kmeans.predict(X_anes5_step)



pca3 = PCA(n_components=3)
pca3.fit(X_all)
X3 = pca3.transform(X_all)
XA_3=pca3.transform(X_anes5_step)


pca3_2 = PCA(n_components=3)
pca3_2.fit(X_all_2)
X3_2 = pca3_2.transform(X_all_2)


fig = plt.figure()
ax = Axes3D(fig)
n=np.where(Y_all==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Anesthesia", edgecolor='k')
n=np.where(Y_all==0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="Resting State", edgecolor='k')
n=np.where(Y_all==2)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="Anesthesia_last 5", edgecolor='k')
#plt.title('Anesthesia 2-Clustering')
#plt.savefig("k_2_cluster",bbox_inches='tight')
plt.title('PCA 3 states (fit all 3)')
plt.savefig("3_States_PCA",bbox_inches='tight')

"""
   SOM
"""
from minisom import MiniSom
#https://github.com/JustGlowing/minisom/blob/master/examples/PoemsAnalysis.ipynb

map_dim = 10
som = MiniSom(map_dim, map_dim, 5460, sigma=1.0, random_seed=1)
#som.random_weights_init(W)
som.train_batch(X_all, num_iteration=len(X_all)*500, verbose=True)


plt.figure(figsize=(10, 10))
# Plotting the response for each pattern in the iris dataset
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
#plt.colorbar()

t = np.zeros(len(Y_all), dtype=int)
t[Y_all == 0] = 0
t[Y_all == 1] = 1

# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for cnt, xx in enumerate(X_all):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
plt.colorbar()
plt.axis([0, 7, 0, 7])
plt.savefig('resulting_images/som_iris.png')
plt.show()




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

# create a 10 x 10 vertex mesh
xx, yy = np.meshgrid(np.linspace(0,9,10), np.linspace(0,9,10))

# create vertices for a rotated mesh (3D rotation matrix)
X =  xx
Y =  yy
Z =  som.distance_map().T

# create some dummy data (20 x 20) for the image
data = som.distance_map().T

# create the figure
fig = plt.figure()

# show the reference image
ax1 = fig.add_subplot(121)
ax1.imshow(data, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')

# show the 3D rotated projection
ax2 = fig.add_subplot(122, projection='3d')
cset = ax2.contourf(X, Y, Z,1000)
for cnt, xx in enumerate(X_all):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    ax2.scatter(w[0], w[1],Z[w[1], w[0]],color=colors[t[cnt]])




plt.colorbar(cset)
plt.show()







#Import the library
import SimpSOM as sps

#Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions.
net = sps.somNet(10, 10, X_all, PBC=True)

#Train the network for 10000 epochs and with initial learning rate of 0.01.
net.train(0.01, 1000)

#Save the weights to file
#net.save('filename_weights')

#Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
#and then according to the distance between each node and its neighbours.
net.nodes_graph(colnum=0)
net.diff_graph()

#Project the datapoints on the new 2D network map.
net.project(raw_data, labels=labels)

#Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(raw_data, type='qthresh')






plt.plot(Z_step_7)
plt.title('Distribution_7')
plt.savefig("All_7_cluster",bbox_inches='tight')

plt.plot(Z_anes5)
plt.title('Distribution_7_prediction')
plt.savefig("Prediction_Recovery",bbox_inches='tight')


np.save('All-7-Cluster',Z_step_7)

All_7=np.load('C:/Users/User/Documents/GitHub/Unsupervised/data/All-7-Cluster.npy')
len(All_7)
len(X_all)

All_7_Anes=All_7[np.where(Y_all==1)]
All_7_Rest=All_7[np.where(Y_all==0)]

All_7_Anes.shape
All_7_Rest.shape

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.hist(All_7_Anes)
ax1.set_title('Anesthesia')
ax2.hist(All_7_Rest)
ax2.set_title('Resting State')

plt.hist(All_7_Anes,label="Anesthesia")
plt.hist(All_7_Rest,label="Rest")

bins = np.arange(10) - 0.5
plt.hist(All_7_Anes, bins,label="Anesthesia")
plt.hist(All_7_Rest, bins,label="Rest")
plt.hist(Z_anes5, bins,label="Anes_last_5")
plt.xticks(range(10))
plt.xlim([-1, 6.9])
plt.show()
plt.legend()

plt.xlim(0,6)
plt.legend()


pca3 = PCA(n_components=3)
pca3.fit(X_all)
X3 = pca3.transform(X_all)

fig = plt.figure()
ax = Axes3D(fig)
plt.title('All PCA 3D_7Groups')
n=np.where(Z_step_7==0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="cluster 0", edgecolor='k')
n=np.where(Z_step_7==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="cluster 1", edgecolor='k')
n=np.where(Z_step_7==2)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="cluster 2", edgecolor='k')
n=np.where(Z_step_7==3)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='orange',label="cluster 3", edgecolor='k')
n=np.where(Z_step_7==4)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='lightblue',label="cluster 4", edgecolor='k')
n=np.where(Z_step_7==5)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='lightgreen',label="cluster 5", edgecolor='k')
n=np.where(Z_step_7==6)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='yellow',label="cluster 6", edgecolor='k')
n=np.where(Z_anes5==4)
ax.scatter(XA_3[n, 0], XA_3[n, 1],XA_3[n, 2], marker='o',color='lightblue',label="cluster 4", edgecolor='k')
ax.legend()
plt.savefig("All_7_PCA",bbox_inches='tight')


"""
    CLUSTER SILOUETTE
"""
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

plt.plot(Z_step_7)

silhouette_avg = silhouette_score(X_all, All_7)


print("For n_clusters =", 7,
      "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X_all, All_7)

fig, (ax1) = plt.subplots(1, 1)
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(X_all) + (7 + 1) * 10])


y_lower = 7
for i in range(7):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[All_7 == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 7)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
