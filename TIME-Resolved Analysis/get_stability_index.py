'''
This Script provides the methods is to calculate the stability index of a K-Means Clustering Solution (propoosed by Lange et al 2004)
'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm


def Stability_Index(X_temp,X_test,P,K,Rep):
    SI=np.zeros([Rep,len(K) ,len(P)])   # Collection of stability index over Repetitions
    x_complete = np.row_stack([X_temp, X_test])  # complete input set for PCA-fit

    for r in range(0,Rep):
        p_i = 0
        print('Repetition '+str(r)+'| '+str(Rep))
        for p in tqdm(P):
            pca = PCA(n_components=p)
            pca.fit(x_complete)
            X_temp_LD = pca.transform(X_temp) # get a low dimension version of X_temp
            X_test_LD = pca.transform(X_test) # and X_test
            k_i=0

            for k in K:
                kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
                kmeans.fit(X_temp_LD)           #fit the classifier on X_template
                S_temp = kmeans.predict(X_test_LD)

                kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
                kmeans.fit(X_test_LD)           #fit the classifier on X_test
                S_test = kmeans.predict(X_test_LD)

                # proportion of disagreeing components in u and v
                SI[r,p_i,k_i]=distance.hamming(S_test,S_temp) # should be already normalized
                k_i=k_i+1

            # increase p iteration by one
            p_i=p_i+1

    SI_M=np.mean(SI,axis=0)
    SI_SD=np.std(SI,axis=0)
    return SI_M , SI_SD




def silhouette_score(X_temp,X_test,P,K,Rep):
    x_complete = np.row_stack([X_temp, X_test])     # complete input set for PCA-fit
    SIL = np.zeros([Rep, len(K), len(P)])           # Collection of silhouette scores over Repetitions

    for r in range(0,Rep):
        p_i = 0
        print('Repetition '+str(r)+'| '+str(Rep))
        for p in tqdm(P):
            pca = PCA(n_components=p)
            pca.fit(x_complete)
            X_complete_LD = pca.transform(x_complete) # and X_test
            k_i=0

            for k in K:
                kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
                kmeans.fit(X_complete_LD)           #fit the classifier on all X_LD
                S_complete = kmeans.predict(X_complete_LD)
                silhouette = silhouette_score(X_complete_LD, S_complete)
                SIL[r,p_i,k_i]=silhouette
                k_i=k_i+1

            # increase p iteration by one
            p_i=p_i+1

    SIL_M = np.mean(SIL, axis=0)
    SIL_SD = np.std(SIL, axis=0)

    return SIL_M , SIL_SD


