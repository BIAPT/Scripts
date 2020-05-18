'''
This Script provides the methods is to calculate the stability index of a K-Means Clustering Solution (propoosed by Lange et al 2004)
'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import random

def compute_stability_index(X,Y_ID,P,K,Rep):

    SI=np.zeros([Rep,len(K) ,len(P)])   # Collection of stability index over Repetitions
    x_complete = X  # complete input set for PCA-fit

    for r in range(0,Rep):
        part = np.unique(Y_ID)
        nr_part = len(part)
        rand = random.sample(range(0, nr_part), 2)  # Choose 2 elements

        X_temp = X[(Y_ID == part[rand[0]]) | (Y_ID == part[rand[1]])]
        X_test = X[(Y_ID != part[rand[0]]) & (Y_ID != part[rand[1]])]
        """
        X_temp = X[(Y_ID == part[rand[0]]) | (Y_ID == part[rand[1]]) | (Y_ID == part[rand[2]]) | (Y_ID == part[rand[3]]) | (
                Y_ID == part[rand[4]])]
        X_test = X[(Y_ID != part[rand[0]]) & (Y_ID != part[rand[1]]) & (Y_ID != part[rand[2]]) & (Y_ID != part[rand[3]]) & (
                Y_ID != part[rand[4]])]
        """
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
                SI[r,k_i,p_i]=distance.hamming(S_test,S_temp) # should be already normalized
                k_i=k_i+1

            # increase p iteration by one
            p_i=p_i+1

    SI_M=np.mean(SI,axis=0)
    SI_SD=np.std(SI,axis=0)
    return SI_M , SI_SD





