import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
import seaborn as sns
from helper_functions import visualize
from helper_functions.General_Information import *
import helper_functions.process_properties as prop
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy

pdf = matplotlib.backends.backend_pdf.PdfPages("../KMC_33_Part_{}_{}_10_1_K6_alpha_heathy_{}.pdf".format(
    value, mode, healthy))

"""
    PCA - all_participants
"""
pca = PCA(n_components=3)
pca.fit(X)
X3 = pca.transform(X)

visualize.plot_pca_results(pdf,X3,Y_out, groupnames)
print("#######  PCA Completed ")

"""
    K_means 7 PC
"""
pca = PCA(n_components=PC)
pca.fit(X)
X7 = pca.transform(X)


for k in KS:
    kmc=KMeans(n_clusters=k, random_state=0,n_init=1000)
    kmc.fit(X7)
    P_kmc=kmc.predict(X7)

    visualize.plot_explained_variance(pdf,pca)
    visualize.plot_clustered_pca(pdf,X3,Y_out,P_kmc,k,groupnames)

    for part in AllPart["Part"]:

        part_cluster = P_kmc[data['ID']==part]
        visualize.plot_pie_and_distribution(pdf, part, part_cluster, k)

    print("#######  K_means with k={} started".format(k))

    for group in partnames:
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
    occurence = prop.calculate_occurence(AllPart,k,P_kmc,data,partnames, groupnames)
    occurence_melt=pd.melt(occurence, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                           value_name="occurence", var_name="State")

    sns.boxplot(x="State", y="occurence", hue="group",
                     data=occurence_melt)
    plt.title('Cluster_Occurence_K-Means')
    pdf.savefig()
    plt.close()

    occ_CRSR = occurence[occurence['ID'].isin(CRSR_ID)]
    if (occ_CRSR['ID']==CRSR_ID).all:
        occ_CRSR.insert (0, "CRSR", list(map(float, CRSR_value)) )

    for s in range(k):
        r, p = scipy.stats.pearsonr(np.array(occ_CRSR[str(s)]), np.array(occ_CRSR['CRSR']))
        r, p = scipy.stats.pearsonr(np.array(occ_CRSR[str(s)]), np.array(occ_CRSR['CRSR']))
        sns.set(style='white', font_scale=1.2)
        g = sns.JointGrid(data=occ_CRSR, x=str(s), y='CRSR')
        g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
        plt.xlabel("state {}".format(s))
        g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
        g.ax_joint.text(0.2, 0.9 , 'r = {0:.2f}, p = {0:.2f}'.format(r,p), fontstyle='italic')
        plt.tight_layout()
        pdf.savefig()
        plt.close()


    """
    Dwell Time
    """
    dwelltime = prop.calculate_dwell_time(AllPart, P_kmc, data, k, partnames, groupnames)
    dwelltime_melt = pd.melt(dwelltime, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                                 value_name="dwell_time", var_name="State")

    sns.boxplot(x="State", y="dwell_time", hue="group",
                     data=dwelltime_melt)
    plt.title('Dwell_Time_K-Means')
    pdf.savefig()
    plt.close()

    for s in range(k):
        # create average connectivity image
        X_conn=np.mean(X.iloc[np.where(P_kmc == s)[0]])
        visualize.plot_connectivity(X_conn, mode)
        pdf.savefig()
        plt.close()

    dwell_CRSR = dwelltime[dwelltime['ID'].isin(CRSR_ID)]
    if (dwell_CRSR['ID']==CRSR_ID).all:
        dwell_CRSR.insert (0, "CRSR", list(map(float, CRSR_value)) )

    for s in range(k):
        r, p = scipy.stats.pearsonr(np.array(dwell_CRSR[str(s)]), np.array(dwell_CRSR['CRSR']))
        r, p = scipy.stats.pearsonr(np.array(dwell_CRSR[str(s)]), np.array(dwell_CRSR['CRSR']))
        sns.set(style='white', font_scale=1.2)
        g = sns.JointGrid(data=dwell_CRSR, x=str(s), y='CRSR')
        g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
        plt.xlabel("state {}".format(s))
        g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
        g.ax_joint.text(0.2, 0.9 , 'r = {0:.2f}, p = {0:.2f}'.format(r,p), fontstyle='italic')
        plt.tight_layout()
        pdf.savefig()
        plt.close()


    """
        Switching Prob
    """
    dynamic = prop.calculate_dynamics(AllPart, P_kmc, data, partnames, groupnames)

    sns.boxplot(x='p_switch', y="group", data=dynamic,
                whis=[0, 100], width=.6)
    sns.stripplot(x='p_switch', y="group", data=dynamic,
                  size=4, color=".3", linewidth=0)

    plt.title("switch state probablilty [%]")
    pdf.savefig()
    plt.close()


    dyn_CRSR = dynamic[dynamic['ID'].isin(CRSR_ID)]
    if (dyn_CRSR['ID']==CRSR_ID).all:
        dyn_CRSR.insert (0, "CRSR", list(map(float, CRSR_value)))

    r, p = scipy.stats.pearsonr(np.array(dyn_CRSR['p_switch']), np.array(dyn_CRSR['CRSR']))
    r, p = scipy.stats.pearsonr(np.array(dyn_CRSR['p_switch']), np.array(dyn_CRSR['CRSR']))
    sns.set(style='white', font_scale=1.2)
    g = sns.JointGrid(data=dyn_CRSR, x='p_switch', y='CRSR')
    g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
    plt.xlabel("switching Probability")
    g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
    g.ax_joint.text(0.2, 0.9 , 'r = {0:.2f}, p = {0:.2f}'.format(r,p), fontstyle='italic')
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    """
        Phase Transition
    """
    # groupwise Phase Transition
    visualize.plot_group_TPM(P_kmc,Y_out,k,pdf,groupnames)

    # individual Phase Transition
    for group in partnames:
        fig, ax = plt.subplots(len(AllPart["{}".format(group)]),1, figsize=(5, 50))
        fig.suptitle('{}; \n {}_Clusters_wholeBrain_alpha'.format(group, k), size=16)
        c = 0
        for part in AllPart["{}".format(group)]:
            part_cluster = P_kmc[data['ID'] == part]
            TPM_part = prop.get_transition_matrix(part_cluster, k)
            sns.heatmap(TPM_part, annot=True, cbar=False, ax=ax[c], fmt='.1g')
            ax[c].set_title('Participant: '+part)
            c +=1
        pdf.savefig(fig)
        plt.close()

    # group averaged Phase Transition
    visualize.plot_group_averaged_TPM(AllPart,P_kmc,Y_out,k,pdf,data,partnames,groupnames)

pdf.close()
print('finished')


