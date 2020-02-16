import scipy.io
import extract_features
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt

"""
################################
        LOAD DATA (not available on gitHub)
################################
"""

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_avg.mat')
data_anes_avg = mat['result_wpli_anes_avg']
X_anes_avg=extract_features.extract_features(data_anes_avg)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_avg.mat')
data_rest_avg = mat['result_wpli_rest_avg']
X_rest_avg=extract_features.extract_features(data_rest_avg)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes_step.mat')
data_anes_step = mat['result_wpli_anes_step']
# extract 5460 features
X_anes_step=extract_features.extract_features(data_anes_step)
#X_anes_step_diff = extract_features.get_difference(X_anes_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_anes5_step.mat')
data_anes5_step = mat['result_wpli_anes5_step']
# extract 5460 features
X_anes5_step=extract_features.extract_features(data_anes5_step)
#X_anes5_step_diff = extract_features.get_difference(X_anes5_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest_step.mat')
data_rest_step = mat['result_wpli_rest_step']
# extract 5460 features
X_rest_step=extract_features.extract_features(data_rest_step)
#X_rest_step_diff = extract_features.get_difference(X_rest_step)

mat = scipy.io.loadmat('data/MDFA05_result_wPLI_rest6_step.mat')
data_rest6_step = mat['result_wpli_rest6_step']
# extract 5460 features
X_rest6_step=extract_features.extract_features(data_rest6_step)
#X_rest6_step_diff = extract_features.get_difference(X_rest6_step)


max_len=min(len(X_rest_step),len(X_anes5_step),len(X_rest_step),len(X_rest6_step))
nr_features=X_rest_step.shape[1]

"""
################################
        CURVE CHARACTERISTICS
################################
"""

''' 1) How much time did the curve spent below 0.05 ? '''

# search for the number of Zero Values in each State
r=[] #rest
r6=[] #rest6
a=[] #anesthesia
a5=[] #pre ROC

for i in range(0,nr_features):
    r.append(len(np.where(X_rest_step[0:max_len,i]<=0.05)[0]))
    r6.append(len(np.where(X_rest6_step[0:max_len,i]<=0.05)[0]))
    a.append(len(np.where(X_anes_step[0:max_len,i]<=0.05)[0]))
    a5.append(len(np.where(X_anes5_step[0:max_len,i]<=0.05)[0]))

r=np.array(r)/max_len
r6=np.array(r6)/max_len
a=np.array(a)/max_len
a5=np.array(a5)/max_len

plt.hist([a,r,a5,r6])
plt.title('time points spent at below 0.05')
plt.xlabel('% of time spent <= 0.05')
plt.ylabel('nr. of features (el-el connection)')
plt.legend(['Anesthesia','Resting State','Anesthesia_5','Resting State 6'])
#plt.savefig('distribution_Zerotime.png')
plt.show()

plt.plot(a)
plt.plot(r)
plt.plot(a5)
plt.plot(r6)
plt.show()

ma=(np.mean(a))
mr=(np.mean(r))
ma5=(np.mean(a5))
mr6=(np.mean(r6))

sa=(np.std(a))
sr=(np.std(r))
sa5=(np.std(a5))
sr6=(np.std(r6))

plt.bar(1,ma, yerr=sa, align='center', ecolor='black', capsize=10)
plt.bar(2,mr, yerr=sr, align='center', ecolor='black', capsize=10)
plt.bar(3,ma5, yerr=sa5, align='center', ecolor='black', capsize=10)
plt.bar(4,mr6, yerr=sr6, align='center', ecolor='black', capsize=10)
plt.title('Percentage of time points over all electrodes below 0.05 ')
plt.ylabel('Percentage ')
plt.legend(['Anesthesia','Resting State','Anesthesia_5','Resting State 6'])
#plt.savefig('mean_percent_Zerotime.png')
plt.show()


plt.bar(range(0,len(a)),a, label='Anesthesia')
plt.bar(range(0,len(r)),r, label='Resting State')
#plt.bar(range(0,len(a5)),a5, label='Anesthesia 5')
#plt.bar(range(0,len(r6)),r6, label='Resting State 6')
#plt.legend(['Anesthesia','Resting State','Anesthesia_5','Resting State 6'])
plt.title('single feature time steps <0.05')
plt.legend()
plt.show()


''' 1) How fast did it oscillate '''

r=[]
a=[]
r6=[]
a5=[]

for i in range(0,nr_features):
    n=np.where(X_rest_step[0:max_len,i]<=0.05)[0]
    n=np.insert(n, 0, 0)
    n=np.append(n,max_len)
    r.append(np.diff(n))

for i in range(0,nr_features):
    n=np.where(X_rest6_step[0:max_len,i]<=0.05)[0]
    n=np.insert(n, 0, 0)
    n=np.append(n,max_len)
    r6.append(np.diff(n))

for i in range(0,nr_features):
    n=np.where(X_anes_step[0:max_len,i]<=0.05)[0]
    n=np.insert(n, 0, 0)
    n=np.append(n,max_len)
    a.append(np.diff(n))

for i in range(0,nr_features):
    n=np.where(X_anes5_step[0:max_len,i]<=0.05)[0]
    n=np.insert(n, 0, 0)
    n=np.append(n,max_len)
    a5.append(np.diff(n))


for i in range(0,nr_features):
    r[i]=np.delete(r[i],[np.where(r[i]<=10)])
    a[i]=np.delete(a[i], [np.where(a[i] <=10)])
    r6[i]=np.delete(r6[i], [np.where(r6[i] <=10)])
    a5[i]=np.delete(a5[i], [np.where(a5[i] <=10)])

r=np.array(r)
r6=np.array(r6)
a=np.array(a)
a5=np.array(a5)

mr=[]
ma=[]
mr6=[]
ma5=[]

for i in range(0,nr_features):
    mr.append(np.mean(r[i]))
    ma.append(np.mean(a[i]))
    mr6.append(np.mean(r6[i]))
    ma5.append(np.mean(a5[i]))

mr=np.array(mr)
mr6=np.array(mr6)
ma=np.array(ma)
ma5=np.array(ma5)

mr[np.where(np.isnan(mr))]=0
ma[np.where(np.isnan(ma))]=0
mr6[np.where(np.isnan(mr6))]=0
ma5[np.where(np.isnan(ma5))]=0

plt.plot(ma, label='Anesthesia')
plt.plot(mr, label='Rest')
plt.plot(ma5, label='Anestheria_last5')
plt.plot(mr6, label='Resting State 6')
plt.title('mean time step difference between wpli<0.05')
plt.legend()
plt.show()


mr=[]
ma=[]
mr6=[]
ma5=[]

for i in range(0,nr_features):
    mr.append(sum(r[i]))
    ma.append(sum(a[i]))
    mr6.append(sum(r6[i]))
    ma5.append(sum(a5[i]))


plt.plot(ma, label='Anesthesia')
plt.plot(mr, label='Rest')
plt.plot(ma5, label='Anestheria_last5')
plt.plot(mr6, label='Resting State 6')
plt.title('length time step difference between wpli<0.05')
plt.legend()
plt.show()


plt.hist([ma,mr,ma5,mr6])
plt.title('number of jumps in the data')
plt.xlabel('number of jumps')
plt.ylabel('nr. of features (el-el connection)')
plt.legend(['Anesthesia','Resting State','Anesthesia_5','Resting State 6'])
#plt.savefig('distribution_jumpnumber.png')
plt.show()


'''ELectrodes of interest'''
eoe=np.where(ma>=1000)

fig, ax = plt.subplots()
ax.bar(2,np.mean(mr), yerr=np.std(mr), align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(1,np.nanmean(ma), yerr=np.nanstd(ma), align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_title('Percentage of time points over all electrodes below 0.05 ')
ax.set_ylabel('Percentage ')
ax.set_xticklabels(' ')
ax.yaxis.grid(True)
ax.legend('r' 'a')


np.where(ma==max(ma))

#i=3747
i=eoe[0][8]
plt.plot(X_rest_step[:,i],label='Resting State')
plt.plot(X_rest6_step[:,i],label='Resting State')
plt.plot(X_anes_step[:,i],label='Anestheia')
plt.plot(X_anes5_step[:,i],label='Anestheia_last5')
plt.legend()




fig, ax = plt.subplots()
ax.bar(2,np.mean(ma[:]), yerr=sr, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.bar(1,ma, yerr=sa, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_title('Percentage of time points over all electrodes below 0.05 ')
ax.set_ylabel('Percentage ')


