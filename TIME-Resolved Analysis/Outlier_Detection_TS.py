import scipy.io
import extract_features
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


mat = scipy.io.loadmat('data/result_wpli_N418_step.mat')
data_step = mat['result_wpli_N418_step']
X_step=extract_features.extract_features(data_step)
X_step=np.array(X_step)
X_step_diff = extract_features.get_difference(X_step)

eventtime = scipy.io.loadmat('data/N418_event_time.mat')
eventtime= eventtime['eventtimes']
eventtime.astype(int)
Y=np.zeros(X_step.shape[0]+100)
Y[eventtime]=1
Y=Y[0:X_step.shape[0]]
len(eventtime[0])

data_step.shape
X_step.shape
Y.shape

e=np.where(Y==1)
ne=np.where(Y==0)

# plot different mean matrixes event, nonevent)
delay=1
plt.imshow(np.mean(data_step[e[0]+delay],0))
plt.imshow(np.mean(data_step[ne],0))

#plot one image to test the data
plt.imshow(X_step[eventtime[0][1:500],:])
#plt.imshow(X_step)
plt.xlabel('feature')
plt.ylabel('timestep')


"""
    PCA PLOT
"""

pca3 = PCA(n_components=3)
pca3.fit(X_step)
X3 = pca3.transform(X_step)

fig = plt.figure()
ax = Axes3D(fig)
n=range(100,5400)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="none", edgecolor='k')
n=np.where(Y==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="event", edgecolor='k')
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
    OUTLIER DETECTION
"""
import pandas as pd
from adtk.data import validate_series

dti = pd.date_range('2020-01-01 00:00:00', periods=len(X_step), freq='S')

df = pd.DataFrame(dti, columns=['date'])
df['data'] = (X_step[:,1000])
df['data2'] = (X_step[:,2000])
df['data4'] = (X_step[:,3000])
df['data5'] = (X_step[:,3500])

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
min_cluster_detector = MinClusterDetector(KMeans(n_clusters=2))
anomalies = min_cluster_detector.fit_detect(df)
plot(df, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');


'''GET ACCURACY'''
plt.plot(Y[anomalies.values])
found_events=len(np.where(Y[anomalies.values]==1)[0])
accuracy=found_events/len(eventtime[0])
accuracy

'''THIS IS SUPER COOL B=) '''
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



