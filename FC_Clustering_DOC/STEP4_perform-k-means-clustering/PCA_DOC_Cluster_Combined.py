import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
import seaborn as sns
from helper_functions import visualize
from helper_functions.General_Information import *
import helper_functions.process_properties as prop

pdf = matplotlib.backends.backend_pdf.PdfPages("Combined_Part_Cluster_Base_wPLI_K5_K6_wholebraind_alpha.pdf")


"""
    PCA - all_participants
"""
pca = PCA(n_components=3)
pca.fit(X)
X3 = pca.transform(X)

visualize.plot_pca_results(pdf,X3,Y_out)

"""
    K_means 7 PC
"""
pca = PCA(n_components=7)
pca.fit(X)
X7 = pca.transform(X)


for k in KS:
    kmc=KMeans(n_clusters=k, random_state=0,n_init=1000)
    kmc.fit(X7)
    P_kmc=kmc.predict(X7)

    visualize.plot_explained_variance(pdf,pca)
    visualize.plot_clustered_pca(pdf,X3,Y_out,P_kmc,k)

    for part in AllPart["Part"]:

        part_cluster = P_kmc[data['ID']==part]
        visualize.plot_pie_and_distribution(pdf, part, part_cluster, k)


    for group in ['Part_nonr', 'Part_reco','Part_heal']:
        fig, ax = plt.subplots(len(AllPart["{}".format(group)]), 1, figsize=(5, 40))
        fig.suptitle('{}; \n {}_Clusters_wholeBrain_alpha'.format(group, k), size=16)
        c = 0
        for part in AllPart["{}".format(group)]:
            #part=AllPart["{}".format(group)][part]
            part_cluster = P_kmc[data['ID'] == part]

            piedata = []
            clusternames = []
            for i in range(k):
                piedata.append(list(part_cluster).count(i))
                clusternames.append('c ' + str(i))

            ax[c].pie(piedata, labels=clusternames, startangle=90)
            ax[c].set_title('Participant: '+part)
            c +=1
        pdf.savefig(fig)
        plt.close()

    """
    Cluster Occurence
    """
    occurence = prop.calculate_occurence(AllPart,k,P_kmc,data)
    occurence_melt=pd.melt(occurence, id_vars=['group'], value_vars=['0','1','2','3','4'],
                           value_name="occurence", var_name="State")

    sns.boxplot(x="State", y="occurence", hue="group",
                     data=occurence_melt)
    plt.title('Cluster_Occurence_K-Means')
    pdf.savefig()
    plt.close()

    """
    Dwell Time
    """
    dwelltime = prop.calculate_dwell_time(AllPart, P_kmc, data, k)
    dwelltime_melt = pd.melt(dwelltime, id_vars=['group'], value_vars=['0', '1', '2', '3', '4'],
                                 value_name="dwell_time", var_name="State")

    sns.boxplot(x="State", y="dwell_time", hue="group",
                     data=dwelltime_melt)
    plt.title('Dwell_Time_K-Means')
    pdf.savefig()
    plt.close()

    for s in range(k):
            # create average connectivity image
            X_conn=np.mean(X.iloc[np.where(P_kmc == s)[0]])
            visualize.plot_connectivity(X_conn)
            pdf.savefig()
            plt.close()

    """
        Switching Prob
    """
    dynamic = prop.calculate_dynamics(AllPart, P_kmc, data)

    sns.boxplot(x='p_switch', y="group", data=dynamic,
                whis=[0, 100], width=.6)
    sns.stripplot(x='p_switch', y="group", data=dynamic,
                  size=4, color=".3", linewidth=0)

    plt.title("switch state probablilty [%]")
    pdf.savefig()
    plt.close()

pdf.close()
print('finished')


