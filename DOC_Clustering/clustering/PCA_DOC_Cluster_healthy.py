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
from clustering import create_image
from scipy import stats
data=pd.read_pickle('data/HEALTHY_Part_WholeBrain_wPLI_10_1_alpha.pickle')
import seaborn as sns

Phase = ['Base']
Part = np.unique(data['ID'])

KS=[5,6]

for p in Phase:
    pdf = matplotlib.backends.backend_pdf.PdfPages("Healthy_Part_Cluster_{}_wPLI_K5_K6_wholebraind_alpha.pdf".format(p))

    data_phase = data.query("Phase=='Base'")
    X = data_phase.iloc[:,4:]

    # Assign outcome
    Y_out=np.zeros(len(X))
    Y_out[data_phase['ID'].isin(Part)] = 3

    """
        PCA - all_participants
    """
    pca = PCA(n_components=3)
    pca.fit(X)
    X3 = pca.transform(X)

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    n= np.where(Y_out==3)
    ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],color='green',label="Healthy Adults")
    plt.title('{}_PCA_allPart_wholeBrain_alpha'.format(p))
    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('{}_PCA_allPart_wholeBrain_alpha'.format(p), size=16)

    ax[0].set_title('PC 0 and 1')
    n = np.where(Y_out == 3)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='green', label="Healthy Adults")

    ax[1].set_title('PC 1 and 2')
    n = np.where(Y_out == 3)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='green', label="Healthy Adults")

    ax[2].set_title('PC 0 and 2')
    n = np.where(Y_out == 3)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='green', label="Healthy Adults")

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
        n = np.where(Y_out==3)[0]
        ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],marker='o',c=P_kmc[n],label="Healthy adults")
        plt.title('{}_{}_Clusters_allPart_wholeBrain_alpha'.format(p,str(k)))
        plt.legend(loc='lower right')
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle('{}_{}_Clusters_allPart_wholeBrain_alpha'.format(p,str(k)), size=16)

        ax[0].set_title('PC 0 and 1')
        n = np.where(Y_out == 3)[0]
        ax[0].scatter(X3[n, 0], X3[n, 1], marker='o', c=P_kmc[n], label="Healthy adults")

        ax[1].set_title('PC 1 and 2')
        n = np.where(Y_out == 3)[0]
        ax[1].scatter(X3[n, 1], X3[n, 2], marker='o', c=P_kmc[n], label="Healthy adults")

        ax[2].set_title('PC 0 and 2')
        n = np.where(Y_out == 3)[0]
        ax[2].scatter(X3[n, 0], X3[n, 2], marker='o', c=P_kmc[n], label="Healthy adults")
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

        fig, ax = plt.subplots(len(Part), 1, figsize=(5, 40))
        fig.suptitle('Healthy adults_{}; \n {}_Clusters_wholeBrain_alpha'.format(p, k), size=16)

        for t in range(0,len(Part)):
            part=Part[t]
            part_cluster = P_kmc[data_phase['ID'] == part]

            piedata = []
            clusternames = []
            for i in range(k):
                piedata.append(list(part_cluster).count(i))
                clusternames.append('c ' + str(i))

            ax[t].pie(piedata, labels=clusternames, startangle=90)
            ax[t].set_title('Participant: '+part)
        pdf.savefig(fig)
        plt.close()

        """
        Cluster Occurence
        """
        occurence = pd.DataFrame(np.zeros((len(Part), k+1)))
        names=["group"]
        for i in range(k):
            names.append(str(i))
        occurence.columns=names

        for s in range(k):
            c = 0
            for t in Part:
                occurence.loc[c,'group'] = "H"
                occurence.loc[c,str(s)] = (len(np.where((P_kmc == s) & (data_phase['ID'] == t))[0]))\
                                          /len(np.where(data_phase['ID'] == t)[0])

                c += 1

        for s in range(k):
            f, ax = plt.subplots(figsize=(7, 6))

            # Plot the orbital period with horizontal boxes
            sns.boxplot(x=str(s), y="group", data=occurence,
                        whis=[0, 100], width=.6, palette="vlag")

            # Add in points to show each observation
            sns.stripplot(x=str(s), y="group", data=occurence,
                          size=4, color=".3", linewidth=0)

            # Tweak the visual presentation
            ax.xaxis.grid(True)
            ax.set(ylabel="")
            sns.despine(trim=True, left=True)
            plt.title("Distribution for state {}".format(s))
            pdf.savefig(f)
            plt.close()

            # create average connectivity image
            X_conn=np.mean(X.iloc[np.where(P_kmc == s)[0]])
            fig = create_image.plot_connectivity(X_conn)
            pdf.savefig()
            plt.close()

        """
        Dwell Time
        """
        dynamic = pd.DataFrame(np.zeros((len(Part), 4)))
        names = ["group","p_stay","p_switch","staytime"]
        dynamic.columns=names

        c=0
        for t in Part:
            dynamic.loc[c, 'group'] = "H"

            part_cluster = P_kmc[data_phase['ID'] == t]

            staytime = []
            stay = len(np.where(np.diff(part_cluster)==0)[0])/len(part_cluster)
            switch = len(np.where(np.diff(part_cluster)!=0)[0])/len(part_cluster)

            s=0
            for l in range(1, len(part_cluster) - 1):
                if part_cluster[l] == part_cluster[l - 1]:
                    s += 1
                if part_cluster[l] != part_cluster[l - 1]:
                    if s>0:
                        staytime.append(s)
                    s = 0

            dynamic.loc[c, "p_switch"] = switch
            dynamic.loc[c, "p_stay"] = stay
            dynamic.loc[c, "staytime"] = np.mean(staytime)
            c += 1

        f, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x='p_switch', y="group", data=dynamic,
                    whis=[0, 100], width=.6, palette="vlag")
        sns.stripplot(x='p_switch', y="group", data=dynamic,
                      size=4, color=".3", linewidth=0)

        # Tweak the visual presentation
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        sns.despine(trim=True, left=True)
        plt.title("switching state")
        pdf.savefig(f)
        plt.close()

        f, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x='p_stay', y="group", data=dynamic,
                    whis=[0, 100], width=.6, palette="vlag")
        sns.stripplot(x='p_stay', y="group", data=dynamic,
                      size=4, color=".3", linewidth=0)

        # Tweak the visual presentation
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        sns.despine(trim=True, left=True)
        plt.title("stay in state ")
        pdf.savefig(f)
        plt.close()

        f, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x='staytime', y="group", data=dynamic,
                    whis=[0, 100], width=.6, palette="vlag")
        sns.stripplot(x='staytime', y="group", data=dynamic,
                      size=4, color=".3", linewidth=0)

        # Tweak the visual presentation
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        sns.despine(trim=True, left=True)
        plt.title("stay time")
        pdf.savefig(f)
        plt.close()

    pdf.close()
    print('finished')

    np.save('kmc','healthy_kmc')


