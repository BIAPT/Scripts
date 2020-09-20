import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
from clustering import create_image
data=pd.read_pickle('data/New_Part_WholeBrain_dPLI_10_1_alpha.pickle')
import seaborn as sns
import pickle

#kmc = np.load('kmc.npy')
#kmc = kmc.astype(KMeans)
#kmc.predict

Phase = ['Base']

Part = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15','S16','S17',
        'S18', 'S19', 'S20', 'S22', 'S23',
        'W03', 'W04', 'W08', 'W22', 'W28','W31', 'W34', 'W36']
Part_nonr = ['S05', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
             'S18', 'S22', 'S23', 'W04', 'W08', 'W28', 'W31', 'W34', 'W36']
Part_reco=['S02', 'S07', 'S09', 'S19', 'S20', 'W03', 'W22']

KS=[5,6]

for p in Phase:
    pdf = matplotlib.backends.backend_pdf.PdfPages("Comparison_Healthy_DOC_Cluster_{}_dPLI_K5_K6_wholebraind_alpha.pdf".format(p))

    data_phase = data.query("Phase=='Base'")
    X = data_phase.iloc[:,4:]

    # Assign outcome
    Y_out=np.zeros(len(X))
    Y_out[data_phase['ID'].isin(Part_reco)] = 1

    """
        PCA - all_participants
    """
    pca = pickle.load(open("healthy_pca3.pkl", "rb"))
    #pca = PCA(n_components=3)
    #pca.fit(X)
    X3 = pca.transform(X)

    """
        K_means 7 PC
    """
    #pca = PCA(n_components=7)
    #pca.fit(X)
    pca = pickle.load(open("healthy_pca7.pkl", "rb"))
    X7 = pca.transform(X)

    for k in KS:
        kmc = pickle.load(open("kmeans_helathy_K{}.pkl".format(k), "rb"))
        P_kmc=kmc.predict(X7)

        # visualize in 3D
        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        n = np.where(Y_out == 1)[0]
        ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], marker='o', c=P_kmc[n], label="Recovered Patients")
        n = np.where(Y_out == 0)[0]
        ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], marker='x', c=P_kmc[n], label="Non-Recovered Patients")
        plt.title('{}_{}_Clusters_allPart_wholeBrain_alpha'.format(p, str(k)))
        plt.legend(loc='lower right')
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle('{}_{}_Clusters_allPart_wholeBrain_alpha'.format(p, str(k)), size=16)

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
            part_cluster = P_kmc[data_phase['ID'] == part]

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle('Part {}_{}; {}_Clusters_wholeBrain_alpha'.format(part, p, k), size=16)

            ax[0].plot(part_cluster)
            ax[0].set_ylim(0, k - 1)
            ax[0].set_title('Part {}_{}; {}_Clusters_wholeBrain_alpha'.format(part, p, k))
            ax[0].set_ylabel('cluaster_Number')
            ax[0].set_xlabel('time')

            piedata = []
            clusternames = []
            for i in range(k):
                piedata.append(list(part_cluster).count(i))
                clusternames.append('cluster ' + str(i))

            ax[1].pie(piedata, labels=clusternames, autopct='%1.1f%%', startangle=90)
            pdf.savefig(fig)
            plt.close()

        fig, ax = plt.subplots(len(Part_nonr), 1, figsize=(5, 40))
        fig.suptitle('Non-recovered_{}; \n {}_Clusters_wholeBrain_alpha'.format(p, k), size=16)

        for t in range(0, len(Part_nonr)):
            part = Part_nonr[t]
            part_cluster = P_kmc[data_phase['ID'] == part]

            piedata = []
            clusternames = []
            for i in range(k):
                piedata.append(list(part_cluster).count(i))
                clusternames.append('c ' + str(i))

            ax[t].pie(piedata, labels=clusternames, startangle=90)
            ax[t].set_title('Participant: ' + part)
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(len(Part_reco), 1, figsize=(5, 15))
        fig.suptitle('Recovered_{}; \n {}_Clusters_wholeBrain_alpha'.format(p, k), size=16)

        for t in range(0, len(Part_reco)):
            part = Part_reco[t]
            part_cluster = P_kmc[data_phase['ID'] == part]

            piedata = []
            clusternames = []
            for i in range(k):
                piedata.append(list(part_cluster).count(i))
                clusternames.append('c ' + str(i))

            ax[t].pie(piedata, labels=clusternames, startangle=90)
            ax[t].set_title('Participant: ' + part)
        pdf.savefig(fig)
        plt.close()

        """
        Cluster Occurence
        """
        occurence = pd.DataFrame(np.zeros((len(Part), k + 1)))
        names = ["group"]
        for i in range(k):
            names.append(str(i))
        occurence.columns = names

        for s in range(k):
            c = 0
            for t in Part_reco:
                occurence.loc[c, 'group'] = "R"
                occurence.loc[c, str(s)] = (len(np.where((P_kmc == s) & (data_phase['ID'] == t))[0])) \
                                           / len(np.where(data_phase['ID'] == t)[0])

                c += 1
            for t in Part_nonr:
                occurence.loc[c, 'group'] = "N"
                occurence.loc[c, str(s)] = (len(np.where((P_kmc == s) & (data_phase['ID'] == t))[0])) \
                                           / len(np.where(data_phase['ID'] == t)[0])

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
            if len(np.where(P_kmc == s)[0]) != 0:
                X_conn = np.mean(X.iloc[np.where(P_kmc == s)[0]])
                fig = create_image.plot_connectivity(X_conn)
                pdf.savefig()
                plt.close()

        """
        Dwell Time
        """
        dynamic = pd.DataFrame(np.zeros((len(Part), 4)))
        names = ["group", "p_stay", "p_switch", "staytime"]
        dynamic.columns = names

        c = 0
        for t in Part:
            if np.isin(t, Part_reco):
                dynamic.loc[c, 'group'] = "R"

            elif np.isin(t, Part_nonr):
                dynamic.loc[c, 'group'] = "N"

            part_cluster = P_kmc[data_phase['ID'] == t]

            staytime = []
            stay = len(np.where(np.diff(part_cluster) == 0)[0]) / len(part_cluster)
            switch = len(np.where(np.diff(part_cluster) != 0)[0]) / len(part_cluster)

            s = 0
            for l in range(1, len(part_cluster) - 1):
                if part_cluster[l] == part_cluster[l - 1]:
                    s += 1
                if part_cluster[l] != part_cluster[l - 1]:
                    if s > 0:
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
