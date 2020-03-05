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
import pandas as pd
import scipy.stats

channels=pd.read_csv('data/MDFA05_channel.txt',sep="#")
channels = channels.iloc[0,0].split(",")
channels=np.array(channels)
#selection=['E24','E11','E124'] #3x Frontal
#selection=['E52','E62','E92'] #3x Parietal
#selection=['E36','E104','Cz'] #3x central

#selection=['E11','E24','E52','E62','E92','E124'] #Frontal-Parietal
#selection=['E11','E24','E36','E104','E124','Cz'] #Frontal-central
#selection=['E36','E52','E62','E92','E104','Cz'] #central-Parietal

selection=['E11','E24','E36','E52','E62','E92','E104','E124','Cz'] #Frontal-central-Parietal





mat = scipy.io.loadmat('data/result_wpli_N418_step.mat')
data_step = mat['result_wpli_N418_step']
data_step_selection=extract_features.extract_single_features(data_step,channels,selection)

X_step_selection=extract_features.extract_features(data_step_selection)
X_step_selection=np.array(X_step_selection)
X_step_selection_diff=extract_features.get_difference(X_step_selection)
X_step=np.mean(X_step_selection_diff,axis=1)

#X_step_s_mean=extract_features.extract_features(data_step_selection,getmean=True)

eventtime = scipy.io.loadmat('data/N418_event_time_con.mat')
eventtime_con= eventtime['coneventtimes']
eventtime = scipy.io.loadmat('data/N418_event_time_incon.mat')
eventtime_incon= eventtime['inconeventtimes']

#eventtime_incon.astype(int)
#Y=np.zeros(X_step.shape[0]+100)
#Y[eventtime_incon]=1

#Y=Y[0:X_step.shape[0]]
#len(np.where(Y==1)[0])

incon_mean=[]
incon_sd=[]
incon_all=[]
for i in range(-20,20):
    Y = np.zeros(X_step.shape[0] + 100)
    Y[eventtime_incon+i] = 1
    Y = Y[0:X_step.shape[0]]
    incon_mean.append(np.mean(X_step[Y==1]))
    incon_sd.append(scipy.stats.sem(X_step[Y==1]))
    incon_all.append(X_step[Y==1])

con_mean=[]
con_sd=[]
con_all=[]
for i in range(-20,20):
    Y = np.zeros(X_step.shape[0] + 100)
    Y[eventtime_con+i] = 1
    Y = Y[0:X_step.shape[0]]
    con_mean.append(np.mean(X_step[Y==1]))
    con_sd.append(scipy.stats.sem(X_step[Y==1]))
    con_all.append(X_step[Y==1])

incon_all=pd.DataFrame(incon_all)

for i in range(1,49):
    plt.plot(range(-20,20),incon_all.iloc[:,i])


incon_all.iloc[:,40].shape

plt.plot(range(-20,20),con_mean)
plt.plot(range(-20,20),incon_mean)
plt.errorbar(range(-20,20),incon_mean, yerr=incon_sd, fmt='.k',elinewidth=0.2);
plt.errorbar(range(-20,20),con_mean, yerr=con_sd, fmt='.k',elinewidth=0.1);
plt.axvline(0, color='k', linestyle='--')
plt.legend(['con', 'incon'])
plt.title('Average Frontal-Central-Parietal')
plt.xlabel('100ms')

plt.plot(range(-20,20),incon_all,linestyle='--')
plt.plot(range(-20,20),incon_mean,linewidth=2)
plt.plot(range(-20,20),con_mean,linewidth=2)
plt.axvline(0, color='k', linestyle='--')
plt.legend(['con', 'incon'])
plt.title('Average C3,C4,Cz')
plt.xlabel('100ms')



data_step.shape
X_step.shape
Y.shape

e=np.where(Y==1)
ne=np.where(Y==0)

# plot different mean matrixes event, nonevent)
delay=0
plt.imshow(np.mean(data_step_selection[e[0]+delay],0))
plt.colorbar()
plt.imshow(np.mean(data_step_selection[ne],0))

"""
    PCA PLOT
"""

pca3 = PCA(n_components=3)
pca3.fit(X_step_selection)
X3 = pca3.transform(X_step_selection)

fig = plt.figure()
ax = Axes3D(fig)
n=np.where(Y==0)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='blue',label="none", edgecolor='k')
n=np.where(Y==1)
ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2], marker='o',color='red',label="event", edgecolor='k')
#plt.title('Anesthesia PCA 3D')
plt.title('Rest PCA 3D')


"""
    OUTLIER DETECTION
"""
import pandas as pd
from adtk.data import validate_series

dti = pd.date_range('2020-01-01 00:00:00', periods=len(X_step), freq='D')
df = pd.DataFrame(dti, columns=['date'])
for i in range(0,X_step.shape[1]):
    df[i] = (X_step[:,i])

df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime')
df.drop(['date'], axis=1, inplace=True)
df.head()
s_train=df

# the same for the label
df = pd.DataFrame(dti, columns=['date'])
df[1] = (Y)
df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime')
df.drop(['date'], axis=1, inplace=True)
df.head()
from adtk.data import to_events
known_anomalies = to_events(df)


from adtk.visualization import plot
plot(s_train,anomaly_true=known_anomalies)
plt.plot(Y)

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
min_cluster_detector = MinClusterDetector(KMeans(n_clusters=3))
anomalies = min_cluster_detector.fit_detect(df)
plot(df, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');


'''GET ACCURACY'''
found_events=len(np.where(Y[anomalies.values]==1)[0])
accuracy=found_events/len(eventtime_incon[0])
accuracy
# !!!! PLOT ROC CURVE !!!!

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




