import scipy.io
import extract_features
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd

"""
LOAD DATA
"""

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
X_anes_step=extract_features.extract_features(data_anes_step)
#X_anes_step_diff = extract_features.get_difference(X_anes_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
X_rest_step=extract_features.extract_features(data_rest_step)
#X_rest_step_diff = extract_features.get_difference(X_rest_step)

X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)
Y_all=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0])),axis=0)

"""
    ISOLATION FOREST
"""
from sklearn.ensemble import IsolationForest
rs=np.random.RandomState(123)
clf = IsolationForest(max_samples=100, random_state=rs,contamination="auto" ,behaviour="new")
clf.fit(X_anes_step)
y_pred_anes = clf.predict(X_anes_step)
y_pred_rest = clf.predict(X_rest_step)

plt.plot(y_pred_anes[:])
plt.plot(y_pred_rest[:])

y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

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


pca3 = PCA(n_components=3)
pca3.fit(X_anes_step)
X3 = pca3.transform(X_anes_step)
X3a = pca3.transform(if_anomalies)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X3[:, 0], X3[:, 1],X3[:, 2], marker='o',color='grey',label="NONE", edgecolor='k')
ax.scatter(X3a[:, 0], X3a[:, 1],X3a[:, 2], marker='o',color='red',label="Anomalies", edgecolor='k')


plt.scatter(df.iloc[:,1],df.iloc[:,2],c='white',s=20,edgecolor='k')
plt.scatter(if_anomalies.iloc[1,:],if_anomalies.iloc[:,1],c='red')
plt.xlabel('Income')
plt.ylabel('Spend_Score')
plt.title('Isolation Forests - Anomalies')




