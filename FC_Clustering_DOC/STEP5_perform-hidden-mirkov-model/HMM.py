import matplotlib
matplotlib.use('Qt5Agg')
from hmmlearn import hmm
import sys
sys.path.append('../')
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from helper_functions import visualize
from helper_functions.General_Information import *
import helper_functions.process_properties as prop
from sklearn.decomposition import PCA

pdf = matplotlib.backends.backend_pdf.PdfPages("HMM_33_Part_Prog_{}_10_1_K6_alpha.pdf".format(mode))


"""
    HMM 7 PC
"""
pca = PCA(n_components=PC)
pca.fit(X)
X7 = pca.transform(X)

for k in KS:
    k
    """
        Hidden Markov Model 
    """
    # create HMM
    scores = []
    models = []

    for i in range(10):
        model = hmm.GaussianHMM(n_components=k, covariance_type="full", n_iter=100)
        model.fit(X7)
        scores.append(model.score(X7))
        models.append(model)

    # select model with highest score
    model = models[np.where(scores == max(scores))[0][0]]

    if model.monitor_.converged == False:
        print('Model not Converged Error')
        break

    P_hmm = model.predict(X7)

    for part in AllPart["Part"]:

        part_cluster = P_hmm[data['ID'] == part]
        visualize.plot_pie_and_distribution(pdf, part, part_cluster, k)

    for group in ['Part_nonr', 'Part_ncmd', 'Part_reco','Part_heal']:
        fig, ax = plt.subplots(len(AllPart["{}".format(group)]), 1, figsize=(5, 40))
        fig.suptitle('{}; \n {}_Clusters_wholeBrain_alpha'.format(group, k), size=16)
        c = 0
        for part in AllPart["{}".format(group)]:
            #part=AllPart["{}".format(group)][part]
            part_cluster = P_hmm[data['ID'] == part]

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
    occurence = prop.calculate_occurence(AllPart,k,P_hmm,data)
    occurence_melt=pd.melt(occurence, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                           value_name="occurence", var_name="State")

    sns.boxplot(x="State", y="occurence", hue="group",
                     data=occurence_melt)
    plt.title('Cluster_Occurence_HMM')
    pdf.savefig()
    plt.close()

    """
    Dwell Time
    """
    dwelltime = prop.calculate_dwell_time(AllPart, P_hmm, data, k)
    dwelltime_melt = pd.melt(dwelltime, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                                 value_name="dwell_time", var_name="State")

    sns.boxplot(x="State", y="dwell_time", hue="group",
                     data=dwelltime_melt)
    plt.title('Dwell_Time_HMM')
    pdf.savefig()
    plt.close()

    for s in range(k):
            # create average connectivity image
            X_conn=np.mean(X.iloc[np.where(P_hmm == s)[0]])
            visualize.plot_connectivity(X_conn, mode= mode)
            pdf.savefig()
            plt.close()

    """
        Switching Prob
    """
    dynamic = prop.calculate_dynamics(AllPart, P_hmm, data)

    sns.boxplot(x='p_switch', y="group", data=dynamic,
                whis=[0, 100], width=.6)
    sns.stripplot(x='p_switch', y="group", data=dynamic,
                  size=4, color=".3", linewidth=0)

    plt.title("switch state probablilty [%]")
    pdf.savefig()
    plt.close()

    """
        Phase Transition
    """
    # groupwise Phase Transition
    visualize.plot_group_TPM(P_hmm,Y_out,k,pdf)

    # individual Phase Transition
    for group in ['Part_nonr', 'Part_ncmd', 'Part_reco','Part_heal']:
        fig, ax = plt.subplots(len(AllPart["{}".format(group)]),1, figsize=(5, 50))
        fig.suptitle('{}; \n {}_Clusters_wholeBrain_alpha'.format(group, k), size=16)
        c = 0
        for part in AllPart["{}".format(group)]:
            part_cluster = P_hmm[data['ID'] == part]
            TPM_part = prop.get_transition_matrix(part_cluster, k)
            sns.heatmap(TPM_part, annot=True, cbar=False, ax=ax[c], fmt='.1g')
            ax[c].set_title('Participant: '+part)
            c +=1
        pdf.savefig(fig)
        plt.close()

    # group averaged Phase Transition
    visualize.plot_group_averaged_TPM(AllPart,P_hmm,Y_out,k,pdf,data)

pdf.close()
print('finished')


