import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import random
import scipy.io
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import stability_measure
from matplotlib import pyplot as plt

'''Import Data'''
data=pd.read_pickle('data/final_wPLI_clustering.pickle')
X=data.iloc[:,4:]
Y_ID=data.iloc[:,1]
Y_St=data.iloc[:,2]
Y_time=data.iloc[:,3]

Part_reco=['WSAS19','WSAS20','WSAS02','WSAS09']
data_reco=data[(Y_ID == 'WSAS19')|(Y_ID == 'WSAS20')|(Y_ID == 'WSAS02')|(Y_ID == 'WSAS09')]
X_reco=data_reco.iloc[:,4:]
Y_ID_reco=data_reco.iloc[:,1]

Part_chron=['WSAS05','WSAS10','WSAS18','WSAS12']
data_chron=data[(Y_ID == 'WSAS05') | (Y_ID == 'WSAS10') | (Y_ID == 'WSAS18') | (Y_ID == 'WSAS12')]
X_chron=data_chron.iloc[:,4:]
Y_ID_chron=data_chron.iloc[:,1]

P=[3,4,5,6,7,8,9,10]     #number of Principal components to iterate
K=[2,3,4,5,6,7,8,9,10]     #number of K-clusters to iterate

SIL=[]

def compute_silhouette_score(X,Y_ID,P,K,Part):
    tmp = np.zeros([len(Part),len(K), len(P)])

    for r in range(0,len(Part)):
        p_i = 0
        X_p=X[Y_ID==Part[r]]

        for p in tqdm(P):
            pca = PCA(n_components=p)
            pca.fit(X_p)
            X_LD = pca.transform(X_p)
            k_i=0

            for k in K:
                kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
                kmeans.fit(X_LD)  # fit the classifier on all X_LD
                S = kmeans.predict(X_LD)
                silhouette = silhouette_score(X_LD, S)
                tmp[r,k_i, p_i] = silhouette
                k_i = k_i + 1

            # increase p iteration by one
            p_i=p_i+1

    return tmp


SIL.append(compute_silhouette_score(X_chron,Y_ID_chron,P,K,Part_chron))
SIL.append(compute_silhouette_score(X_reco,Y_ID_reco,P,K,Part_reco))



fig,a =  plt.subplots(4,2)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')

im=a[0][0].imshow(np.transpose(SIL[0][0]))
a[0][0].set_title('Silhouette Score:'+Part_chron[0])
plt.colorbar(im,ax=a[0,0])
im=a[1][0].imshow(np.transpose(SIL[0][1]))
a[1][0].set_title('Silhouette Score:'+Part_chron[1])
plt.colorbar(im,ax=a[0,1])
im=a[2][0].imshow(np.transpose(SIL[0][2]))
a[2][0].set_title('Silhouette Score:'+Part_chron[2])
plt.colorbar(im,ax=a[1,0])
im=a[3][0].imshow(np.transpose(SIL[0][3]))
a[3][0].set_title('Silhouette Score:'+Part_chron[3])
plt.colorbar(im,ax=a[1,1])

im=a[0][1].imshow(np.transpose(SIL[1][0]))
a[0][1].set_title('Silhouette Score:'+Part_reco[0])
plt.colorbar(im,ax=a[2,0])
im=a[1][1].imshow(np.transpose(SIL[1][1]))
a[1][1].set_title('Silhouette Score:'+Part_reco[1])
plt.colorbar(im,ax=a[2,1])
im=a[2][1].imshow(np.transpose(SIL[1][2]))
a[2][1].set_title('Silhouette Score:'+Part_reco[2])
plt.colorbar(im,ax=a[3,0])
im=a[3][1].imshow(np.transpose(SIL[1][3]))
a[3][1].set_title('Silhouette Score:'+Part_reco[3])
plt.colorbar(im,ax=a[3,1])


plt.plot(SIL[0][0][:,7],color='red')
plt.plot(SIL[0][1][:,7],color='red')
plt.plot(SIL[0][2][:,7],color='red')
plt.plot(SIL[0][3][:,7],color='red')
#plt.legend([Part_chron[0],Part_chron[1],Part_chron[2],Part_chron[3]])

plt.plot(SIL[1][0][:,7],color='blue')
plt.plot(SIL[1][1][:,7],color='blue')
plt.plot(SIL[1][2][:,7],color='blue')
plt.plot(SIL[1][3][:,7],color='blue')
#plt.legend([Part_reco[0],Part_reco[1],Part_reco[2],Part_reco[3]])


"""
Silhouette Score
"""
P=[3,4,5,6,7,8,9,10]     #number of Principal components to iterate
K=[2,3,4,5,6,7,8,9,10]     #number of K-clusters to iterate
Rep=2         #number of Repetitions (Mean at the end)

data_19=data[(Y_ID == 'WSAS19')]
X_19=data_19.iloc[:,4:]
[SIL_M_19 ,SIL_SD_19] = stability_measure.compute_silhouette_score(X_19, P, K, Rep)
data_09=data[(Y_ID == 'WSAS09')]
X_09=data_09.iloc[:,4:]
[SIL_M_09 ,SIL_SD_09] = stability_measure.compute_silhouette_score(X_09, P, K, Rep)
data_05=data[(Y_ID == 'WSAS05')]
X_05=data_05.iloc[:,4:]
data_02=data[(Y_ID == 'WSAS02')]
X_02=data_02.iloc[:,4:]
data_20=data[(Y_ID == 'WSAS20')]
X_20=data_20.iloc[:,4:]

fig,a =  plt.subplots(2,2)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')

im=a[0][0].imshow(np.transpose(SIL_M_19))
a[0][0].set_title('Silhouette Score Mean: Recovered patient 19')
plt.colorbar(im,ax=a[0,0])
im=a[0][1].imshow(np.transpose(SIL_SD_19))
a[0][1].set_title('Silhouette Score SD: Recovered patient 19')
plt.colorbar(im,ax=a[0,1])
im=a[1][0].imshow(np.transpose(SIL_M_09))
a[1][0].set_title('Silhouette Score Mean: Acute patient 09 ')
plt.colorbar(im,ax=a[1,0])
im=a[1][1].imshow(np.transpose(SIL_SD_09))
a[1][1].set_title('Silhouette Score SD: Acute patient 09')
plt.colorbar(im,ax=a[1,1])









[SIL_M_acute ,SIL_SD_acute] = stability_measure.compute_silhouette_score(X_acute, P, K, Rep)
[SIL_M_reco ,SIL_SD_reco] = stability_measure.compute_silhouette_score(X_reco, P, K, Rep)


fig,a =  plt.subplots(2,2)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')
im=a[0][0].imshow(np.transpose(SIL_M_reco))
a[0][0].set_title('Silhouette Score Mean: Recovered patients')
plt.colorbar(im,ax=a[0,0])
im=a[0][1].imshow(np.transpose(SIL_SD_reco))
a[0][1].set_title('Silhouette Score SD: Recovered patients')
plt.colorbar(im,ax=a[0,1])
im=a[1][0].imshow(np.transpose(SIL_M_acute))
a[1][0].set_title('Silhouette Score Mean: Acute patients')
plt.colorbar(im,ax=a[1,0])
im=a[1][1].imshow(np.transpose(SIL_SD_acute))
a[1][1].set_title('Silhouette Score SD: Acute patients')
plt.colorbar(im,ax=a[1,1])


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.metrics import silhouette_score


SIL_19=[]
SIL_09=[]
SIL_05=[]
SIL_02=[]
SIL_20=[]

for k in K:
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
    kmeans.fit(X_20)  # fit the classifier on all X_LD
    S_complete = kmeans.predict(X_20)
    silhouette = silhouette_score(X_20, S_complete)
    SIL_20.append(silhouette)

plt.plot(SIL_19)
plt.plot(SIL_09)
plt.plot(SIL_05)
plt.plot(SIL_02)
plt.plot(SIL_20)
plt.legend(['19','09','05','02','20'])