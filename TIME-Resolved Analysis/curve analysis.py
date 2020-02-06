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

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes_avg.mat')
data_anes_avg = mat['result_wpli_anes_avg']
X_rest_avg=extract_features.extract_features(data_anes_avg)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest_avg.mat')
data_rest_avg = mat['result_wpli_rest_avg']
X_rest_avg=extract_features.extract_features(data_rest_avg)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
X_anes_step=extract_features.extract_features(data_anes_step)
X_anes_step_diff = extract_features.get_difference(X_anes_step)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_anes5_step.mat')
data_anes5_step = mat['result_wpli_anes5_step']
X_anes5_step=extract_features.extract_features(data_anes5_step)
X_anes5_step_diff = extract_features.get_difference(X_anes5_step)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
X_rest_step=extract_features.extract_features(data_rest_step)
X_rest_step_diff = extract_features.get_difference(X_rest_step)

mat = scipy.io.loadmat('C:/Users/User/Documents/GitHub/Unsupervised/data/MDFA05_result_wPLI_rest6_step.mat')
data_rest6_step = mat['result_wpli_rest6_step']
X_rest6_step=extract_features.extract_features(data_rest6_step)


X_all= np.concatenate((X_anes_step,X_rest_step),axis=0)
Y_all=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0])),axis=0)

X_all_2= np.concatenate((X_anes_step,X_rest_step,X_anes5_step),axis=0)
Y_all_2=np.concatenate((np.ones(X_anes_step.shape[0]),np.zeros(X_rest_step.shape[0]), np.repeat(2, X_anes5_step.shape[0])),axis=0)


"""
    Curve Characteristic
"""

r=[]
a=[]

for i in range(0,X_rest_step.shape[1]):
    r.append(len(np.where(X_rest_step[0:2830,i]<=0.05)[0]))
for i in range(0,X_anes_step.shape[1]):
    a.append(len(np.where(X_anes_step[0:2830,i]<=0.05)[0]))

ma=(np.mean(a))
mr=(np.mean(r))
sa=(np.std(a))
sr=(np.std(r))

fig, ax = plt.subplots()
ax.bar(2,mr, yerr=sr, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(1,ma, yerr=sa, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_title('Percentage of time points over all electrodes below 0.05 ')
ax.set_ylabel('Percentage ')
ax.set_xticklabels(' ')
ax.yaxis.grid(True)
ax.legend('r' 'a')

plt.plot(a, label='Anesthesia')
plt.plot(r, label='Rest')
plt.title('sum time steps <0.05')
plt.legend()

'''
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.graphics import tsaplots
plt.plot(X_rest_step[:,1])
# Display the autocorrelation plot of your time series
fig = tsaplots.plot_acf(X_rest_step[:,100], lags=2800)
# Show plot
plt.show()
'''

r=[]
a=[]
m=[]
s=[]

for i in range(0,X_rest_step.shape[1]):
    n=np.where(X_rest_step[0:2830,i]<=0.05)[0]
    n=np.insert(n, 0, 0)
    n=np.append(n,2830)
    r.append(np.diff(n))

for i in range(0,X_rest_step.shape[1]):
    n=np.where(X_rest6_step[0:2830,i]<=0.05)[0]
    n=np.insert(n, 0, 0)
    n=np.append(n,2830)
    s.append(np.diff(n))

for i in range(0,X_rest_step.shape[1]):
    n=np.where(X_anes5_step[0:2830,i]<=0.05)[0]
    n=np.insert(n, 0, 0)
    n=np.append(n,2830)
    m.append(np.diff(n))

for i in range(0,X_anes_step.shape[1]):
    n = np.where(X_anes_step[0:2830, i] <= 0.05)[0]
    n = np.insert(n, 0, 0)
    n = np.append(n, 2830)
    a.append(np.diff(n))

for i in range(0,X_anes_step.shape[1]):
    r[i]=np.delete(r[i],[np.where(r[i]<=5)])
    a[i]=np.delete(a[i], [np.where(a[i] <=5)])
    m[i]=np.delete(m[i], [np.where(m[i] <=5)])
    s[i]=np.delete(s[i], [np.where(s[i] <=5)])

mr=[]
ma=[]
mm=[]
ms=[]

for i in range(0,X_rest_step.shape[1]):
    mr.append(np.mean(r[i]))
    ma.append(np.mean(a[i]))
    mm.append(np.mean(m[i]))
    ms.append(np.mean(s[i]))


plt.plot(ma, label='Anesthesia')
plt.plot(mr, label='Rest')
plt.plot(mm, label='Anestheria_last5')
plt.plot(ms, label='Resting State 6')
plt.title('mean time step difference between wpli<0.05')
plt.legend()

ma=np.array(ma)
eoe=np.where(ma>=500)

fig, ax = plt.subplots()
ax.bar(2,np.mean(mr), yerr=np.std(mr), align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(1,np.nanmean(ma), yerr=np.nanstd(ma), align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_title('Percentage of time points over all electrodes below 0.05 ')
ax.set_ylabel('Percentage ')
ax.set_xticklabels(' ')
ax.yaxis.grid(True)
ax.legend('r' 'a')


np.where(ma==max(ma))

i=400
plt.plot(X_rest_step[:,i],label='Resting State')
plt.plot(X_anes_step[:,i],label='Anestheia')
plt.plot(X_anes5_step[:,i],label='Anestheia_last5')
plt.legend()




fig, ax = plt.subplots()
ax.bar(2,np.mean(ma[:]), yerr=sr, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(1,ma, yerr=sa, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_title('Percentage of time points over all electrodes below 0.05 ')
ax.set_ylabel('Percentage ')


