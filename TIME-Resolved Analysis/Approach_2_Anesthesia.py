import scipy.io
import extract_features
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes_avg.mat')
data_anes_avg = mat['result_wpli_anes_avg']
data_anes_avg.shape
X_rest_avg=extract_features.extract_features(data_anes_avg)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest_avg.mat')
data_rest_avg = mat['result_wpli_rest_avg']
X_rest_avg=extract_features.extract_features(data_rest_avg)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
data_anes_step.shape
X_anes_step=extract_features.extract_features(data_anes_step)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
X_rest_step=extract_features.extract_features(data_rest_step)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes5_step.mat')
data_anes5_step = mat['result_wpli_anes5_step']
X_anes5_step=extract_features.extract_features(data_anes5_step)
X_anes5_step_diff = extract_features.get_difference(X_anes5_step)

plt.imshow(data_anes_avg)

X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)
Y_all=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0])),axis=0)

X_all_2= np.concatenate((X_anes_step,X_rest_step,X_anes5_step),axis=0)
Y_all_2=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0]), np.repeat(2, X_anes5_step.shape[0])),axis=0)

"""
    Hierarchical Clustering
"""
cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
# ward minimizes variance between clusters

cluster.fit_predict(X_all_2)
print(cluster.labels_)

plt.plot(cluster.labels_)
plt.title('Hierarchical Clustering Eucledian Distance')
plt.xlim(0, 6000)

fig= plt.figure()

# run PCA to Visualize
pca3 = PCA(n_components=3)
pca3.fit(X_all_2)
X3 = pca3.transform(X_all_2)




fig = plt.figure()
ax = Axes3D(fig)
plt.title('Hierarchical All PCA 3D_7Groups')
n=np.where(cluster.labels_==0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="cluster 0", edgecolor='k')
n=np.where(cluster.labels_==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="cluster 1", edgecolor='k')
n=np.where(cluster.labels_==2)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="cluster 2", edgecolor='k')
n=np.where(cluster.labels_==3)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='orange',label="cluster 3", edgecolor='k')
n=np.where(cluster.labels_==4)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='lightblue',label="cluster 4", edgecolor='k')
n=np.where(cluster.labels_==5)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='lightgreen',label="cluster 5", edgecolor='k')
n=np.where(cluster.labels_==6)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='yellow',label="cluster 6", edgecolor='k')
ax.legend()
plt.savefig("Hierarchical_All_7_PCA",bbox_inches='tight')


All_7_Anes=cluster.labels_[np.where(Y_all_2==1)]
All_7_Rest=cluster.labels_[np.where(Y_all_2==0)]
All_7_Anes5=cluster.labels_[np.where(Y_all_2==2)]


All_7_Anes.shape
All_7_Rest.shape
All_7_Anes5.shape

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.hist(All_7_Anes)
ax1.set_title('Anesthesia')
ax2.hist(All_7_Rest)
ax2.set_title('Resting State')

plt.hist(All_7_Anes,label="Anesthesia")
plt.hist(All_7_Rest,label="Rest")

bins = np.arange(10) - 0.5
plt.hist(All_7_Anes5, bins,label="Anesthesia_last5")
plt.hist(All_7_Anes, bins,label="Anesthesia")
plt.hist(All_7_Rest, bins,label="Rest")
plt.xticks(range(10))
plt.xlim([-1, 6.9])
plt.legend()
plt.show()

plt.xlim(0,6)
plt.legend()













"""
    K-MEANS
"""

wcss_anes = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=500, random_state=0)
    kmeans.fit(X_anes_step)
    wcss_anes.append(kmeans.inertia_)
    print(str(i))

plt.figure()
plt.plot(range(1, 10), wcss_anes,color="red")
plt.title('Elbow Method Anesthesia Data')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.savefig("ellbow_anes.png",bbox_inches='tight')

"""
    PCA PLOT
"""

pca3 = PCA(n_components=3)
pca3.fit(X_anes_step)
X3 = pca3.transform(X_anes_step)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X3[:, 0], X3[:, 1],X3[:, 2], marker='o',color='red',label="Anesthesia", edgecolor='k')
#plt.title('Anesthesia PCA 3D')
plt.title('Rest PCA 3D')

"""
    PCA ANALYSIS
"""

pca = PCA().fit(X_anes_step)
plt.plot(np.cumsum(pca.explained_variance_ratio_)[0:50])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
#plt.savefig("pca_anes_4.png",bbox_inches='tight')


"""
    Cluster ANALYSIS- 3d visualization
"""
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=500, random_state=0)
kmeans.fit(X_anes_step)
Z_step_2 = kmeans.predict(X_anes_step)

fig = plt.figure()
ax = Axes3D(fig)
n=np.where(Z_step_2==0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="cluster 1", edgecolor='k')
n=np.where(Z_step_2==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="cluster 2", edgecolor='k')
#plt.title('Anesthesia 2-Clustering')
#plt.savefig("k_2_cluster",bbox_inches='tight')
plt.title('Rest 2-Clustering')
plt.savefig("k_2_cluster_rest",bbox_inches='tight')


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=500, random_state=0)
kmeans.fit(X_anes_step)
Z_step_3 = kmeans.predict(X_anes_step)

fig = plt.figure()
ax = Axes3D(fig)
n=np.where(Z_step_3==0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="cluster 1", edgecolor='k')
n=np.where(Z_step_3==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="cluster 2", edgecolor='k')
n=np.where(Z_step_3==2)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="cluster 3", edgecolor='k')
#plt.title('Anesthesia 3-Clustering')
#plt.savefig("k_3_cluster",bbox_inches='tight')
plt.title('Rest 3-Clustering')
plt.savefig("k_3_cluster_rest",bbox_inches='tight')


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=500, random_state=0)
kmeans.fit(X_anes_step)
Z_step_4 = kmeans.predict(X_anes_step)


fig = plt.figure()
ax = Axes3D(fig)
n=np.where(Z_step_4==0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="cluster 1", edgecolor='k')
n=np.where(Z_step_4==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='green',label="cluster 2", edgecolor='k')
n=np.where(Z_step_4==2)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="cluster 3", edgecolor='k')
n=np.where(Z_step_4==3)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='orange',label="cluster 4", edgecolor='k')
#plt.title('Anesthesia 4-Clustering')
#plt.savefig("k_4_cluster",bbox_inches='tight')
plt.title('Rest 4-Clustering')
plt.savefig("k_4_cluster_rest",bbox_inches='tight')


X_anes_k_results=np.zeros((Z_step_2.shape[0],3))
X_anes_k_results[:,0] = Z_step_2
X_anes_k_results[:,1] = Z_step_3
X_anes_k_results[:,2] = Z_step_4

#np.save('Anesthesia_k-Cluster',X_anes_k_results)
np.save('Rest_k-Cluster',X_anes_k_results)
cluster_data=np.load('C:/Users/User/Documents/GitHub/Unsupervised/data/Anesthesia_k-Cluster.npy')

plt.plot(Z_step_4)
plt.title('Distribution_4')
plt.savefig("Dist_4_rest_cluster",bbox_inches='tight')


plt.figure()
plt.plot(X_anes_step[:,1:1000])

plt.imshow(X_anes_step)


"""
    OUTLIER DETECTION
"""

n=2422
ax1=plt.subplot(2, 1, 1)
ax1.title.set_text('Anesthesia feature  ' + str(n))
ax1.plot(X_anes_step[:,n])
ax2=plt.subplot(2, 1, 2)
ax2.title.set_text('resting State  feature  ' + str(n))
ax2.plot(X_rest_step[:,n])


n=2128
ax1=plt.subplot(2, 1, 1)
ax1.title.set_text('Anesthesia diff feature  ' + str(n))
ax1.plot(X_anes_step_diff[:,n])
ax2=plt.subplot(2, 1, 2)
ax2.title.set_text('resting State diff feature  ' + str(n))
ax2.plot(X_rest_step_diff[:,n])



import pandas as pd
import datetime
from adtk.data import validate_series

dti = pd.date_range('2020-01-01 00:00:00', periods=len(X_rest_step), freq='S')

df = pd.DataFrame(dti, columns=['date'])
df['data'] = (X_rest_step[:,1000])
df['data2'] = (X_rest_step[:,2000])

df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime')
df.drop(['date'], axis=1, inplace=True)
df.head()

s_train = validate_series(df)
from adtk.visualization import plot
plot(s_train)

from adtk.detector import SeasonalAD
seasonal_ad = SeasonalAD()
anomalies = seasonal_ad.fit_detect(s_train)
plot(s_train, anomaly_pred=anomalies, ap_color='red', ap_marker_on_curve=True)

from adtk.detector import LevelShiftAD
levelshift_ad = LevelShiftAD()
anomalies = levelshift_ad.fit_detect(s_train)
plot(s_train, anomaly_pred=anomalies, ap_color='red', ap_marker_on_curve=True)

from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
min_cluster_detector = MinClusterDetector(KMeans(n_clusters=5))
anomalies = min_cluster_detector.fit_detect(df)
plot(df, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');

from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor
outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
anomalies = outlier_detector.fit_detect(df)
plot(df, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');

from adtk.detector import RegressionAD
from sklearn.linear_model import LinearRegression
regression_ad = RegressionAD(regressor=LinearRegression(), target="data2", c=3.0)
anomalies = regression_ad.fit_detect(df)
plot(df, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');


from adtk.transformer import RollingAggregate
s_transformed = RollingAggregate(agg='count', window=5).transform(df.iloc[:,1])
plot(s_transformed, ts_linewidth=2, ts_markersize=3);



from sklearn.ensemble import IsolationForest
rs=np.random.RandomState(123)
clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1)
clf.fit(X_anes_step)

if_scores = clf.decision_function(X_anes_step)
if_anomalies=clf.predict(X_anes_step)
if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])
if_anomalies=X_anes_step[if_anomalies==1];

plt.figure(figsize=(12,8))
plt.hist(if_scores);
plt.title('Histogram of Avg Anomaly Scores: Lower => More Anomalous');

cmap=np.array(['white','red'])

fig = plt.figure()
ax = Axes3D(fig)
n = np.where(if_anomalies == 0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="Anomalies", edgecolor='k')
n = np.where(if_anomalies != 0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='grey',label="NONE", edgecolor='k')


plt.scatter(df.iloc[:,1],df.iloc[:,2],c='white',s=20,edgecolor='k')
plt.scatter(if_anomalies.iloc[1,:],if_anomalies.iloc[:,1],c='red')
plt.xlabel('Income')
plt.ylabel('Spend_Score')
plt.title('Isolation Forests - Anomalies')



