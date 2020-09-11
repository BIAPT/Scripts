import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
from sklearn import preprocessing

data=pd.read_pickle('data/WholeBrain_dPLI_10_1_alpha.pickle')

Phase=['Both']

Part = ['13', '18', '05', '11', '22', '12', '10', '09', '19', '02', '20']
Part_nonr = ['13', '18', '05', '11', '22', '12', '10']
Part_reco=['02', '09', '19', '20']

#KS=[3,4]
KS=[5,6]


for p in Phase:
    pdf = matplotlib.backends.backend_pdf.PdfPages("Scaled_Both_Phases_Part_Cluster_{}_dPLI_12C_K5_K6_wholebraind_alpha.pdf".format(p))

    if p=='Both':
        data_phase=data.query("Phase!='Reco'")
        X=data_phase.iloc[:,4:]
    else:
        data_phase=data.query("Phase=='{}'".format(p))
        X=data_phase.iloc[:,4:]

    X=preprocessing.scale(X)


    # Assign outcome
    Y_out=np.zeros(len(X))
    Y_out[data_phase['ID'].isin(Part_reco)] = 1

    """
        PCA - all_participants
    """
    pca = PCA(n_components=3)
    pca.fit(X)
    X3 = pca.transform(X)

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    n= np.where(Y_out==1)
    ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],color='red',label="Recovered Patients")
    n= np.where(Y_out==0)
    ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],color='blue',label="Non-Recovered Patients")
    plt.title('{}_PCA_allPart_wholeBrain_alpha'.format(p))
    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('{}_PCA_allPart_wholeBrain_alpha'.format(p), size=16)

    ax[0].set_title('PC 0 and 1')
    n = np.where(Y_out == 1)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='red', label="Recovered Patients")
    n = np.where(Y_out == 0)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='blue', label="Non-Recovered Patients")

    ax[1].set_title('PC 1 and 2')
    n = np.where(Y_out == 1)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='red', label="Recovered Patients")
    n = np.where(Y_out == 0)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='blue', label="Non-Recovered Patients")

    ax[2].set_title('PC 0 and 2')
    n = np.where(Y_out == 1)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='red', label="Recovered Patients")
    n = np.where(Y_out == 0)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='blue', label="Non-Recovered Patients")
    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()

    """
        K_means 7 PC
    """
    pca = PCA(n_components=7)
    pca.fit(X)
    X7 = pca.transform(X)

    # PLot explained Variance
    fig = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('{}_Explained_Variance_allPart_wholeBrain_alpha'.format(p))
    pdf.savefig(fig)
    plt.close()



    for k in KS:
        kmc=KMeans(n_clusters=k, random_state=0,n_init=1000)
        kmc.fit(X7)
        P_kmc=kmc.predict(X7)

        # visualize in 3D
        fig = plt.figure(figsize=(6,6))
        ax = Axes3D(fig)
        n = np.where(Y_out==1)[0]
        ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],marker='o',c=P_kmc[n],label="Recovered Patients")
        n= np.where(Y_out==0)[0]
        ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],marker='x',c=P_kmc[n],label="Non-Recovered Patients")
        plt.title('{}_{}_Clusters_allPart_wholeBrain_alpha'.format(p,str(k)))
        plt.legend(loc='lower right')
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle('{}_{}_Clusters_allPart_wholeBrain_alpha'.format(p,str(k)), size=16)

        ax[0].set_title('PC 0 and 1')
        n = np.where(Y_out == 1)[0]
        ax[0].scatter(X3[n, 0], X3[n, 1], marker='o', c=P_kmc[n], label="Recovered Patients")
        n = np.where(Y_out == 0)[0]
        ax[0].scatter(X3[n, 0], X3[n, 1], marker='x', c=P_kmc[n], label="Non-Recovered Patients")

        ax[1].set_title('PC 1 and 2')
        n = np.where(Y_out == 1)[0]
        ax[1].scatter(X3[n, 1], X3[n, 2], marker='o', c=P_kmc[n], label="Recovered Patients")
        n = np.where(Y_out == 0)[0]
        ax[1].scatter(X3[n, 1], X3[n, 2], marker='x', c=P_kmc[n], label="Non-Recovered Patients")

        ax[2].set_title('PC 0 and 2')
        n = np.where(Y_out == 1)[0]
        ax[2].scatter(X3[n, 0], X3[n, 2], marker='o', c=P_kmc[n], label="Recovered Patients")
        n = np.where(Y_out == 0)[0]
        ax[2].scatter(X3[n, 0], X3[n, 2], marker='x', c=P_kmc[n], label="Non-Recovered Patients")
        plt.legend(loc='lower right')
        pdf.savefig(fig)
        plt.close()

        for part in Part:
            part_cluster = P_kmc[data_phase['ID']==part]

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle('Part {}_{}; {}_Clusters_wholeBrain_alpha'.format(part,p,k), size=16)

            ax[0].plot(part_cluster)
            ax[0].set_ylim(0,k-1)
            ax[0].set_title('Part {}_{}; {}_Clusters_wholeBrain_alpha'.format(part,p,k))
            ax[0].set_ylabel('cluaster_Number')
            ax[0].set_xlabel('time')

            piedata = []
            clusternames=[]
            for i in range(k):
                piedata.append(list(part_cluster).count(i))
                clusternames.append('cluster '+str(i))

            ax[1].pie(piedata, labels=clusternames ,autopct='%1.1f%%',startangle=90)
            pdf.savefig(fig)
            plt.close()

        fig, ax = plt.subplots(len(Part_nonr), 2, figsize=(5, 40))
        fig.suptitle('Non-recovered_{}; \n {}_Clusters_wholeBrain_alpha'.format(p, k), size=16)

        for t in range(0,len(Part_nonr)):
            part=Part_nonr[t]
            part_cluster_Base = P_kmc[(data_phase['ID'] == part) & (data_phase['Phase'] == 'Base')]
            part_cluster_Anes = P_kmc[(data_phase['ID'] == part) & (data_phase['Phase'] == 'Anes')]

            piedata_B = []
            clusternames_B = []
            for i in range(k):
                piedata_B.append(list(part_cluster_Base).count(i))
                clusternames_B.append('c ' + str(i))

            piedata_A = []
            clusternames_A = []
            for i in range(k):
                piedata_A.append(list(part_cluster_Anes).count(i))
                clusternames_A.append('c ' + str(i))

            ax[t][0].pie(piedata_B, labels=clusternames_B, startangle=90)
            ax[t][1].pie(piedata_A, labels=clusternames_A, startangle=90)
            ax[t][1].set_title('Participant: '+part)
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(len(Part_reco), 2, figsize=(5, 15))
        fig.suptitle('Recovered_{}; \n {}_Clusters_wholeBrain_alpha'.format(p, k), size=16)

        for t in range(0,len(Part_reco)):
            part=Part_reco[t]
            part_cluster_Base = P_kmc[(data_phase['ID'] == part) & (data_phase['Phase'] == 'Base')]
            part_cluster_Anes = P_kmc[(data_phase['ID'] == part) & (data_phase['Phase'] == 'Anes')]

            piedata_B = []
            clusternames_B = []
            for i in range(k):
                piedata_B.append(list(part_cluster_Base).count(i))
                clusternames_B.append('c ' + str(i))

            piedata_A = []
            clusternames_A = []
            for i in range(k):
                piedata_A.append(list(part_cluster_Anes).count(i))
                clusternames_A.append('c ' + str(i))

            ax[t][0].pie(piedata_B, labels=clusternames_B, startangle=90)
            ax[t][1].pie(piedata_A, labels=clusternames_A, startangle=90)
            ax[t][1].set_title('Participant: ' + part)
        pdf.savefig(fig)
        plt.close()

    pdf.close()
    print('finished')


