import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
import seaborn as sns
from helper_functions import visualize
from helper_functions.General_Information import *
import helper_functions.process_properties as prop

pdf = matplotlib.backends.backend_pdf.PdfPages("HMM_Combined_Part_Cluster_Base_wPLI_K5_K6_wholebraind_alpha.pdf")


for k in KS:
    """
        Hidden Markov Model 
    """
    # create HMM
    hmm = hmm.GaussianHMM(n_components=k, covariance_type="full", n_iter=100)
    hmm.fit(X)
    hmm.score(X)

    P_hmm = hmm.predict(X)

    #tpm = prop.get_transition_matrix()

    for part in AllPart["Part"]:

        part_cluster = P_hmm[data['ID'] == part]
        visualize.plot_pie_and_distribution(pdf, part, part_cluster, k)

    for group in ['Part_nonr', 'Part_reco','Part_heal']:
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
    dwelltime = prop.calculate_dwell_time(AllPart, P_hmm, data, k)
    dwelltime_melt = pd.melt(dwelltime, id_vars=['group'], value_vars=['0', '1', '2', '3', '4'],
                                 value_name="dwell_time", var_name="State")

    sns.boxplot(x="State", y="dwell_time", hue="group",
                     data=dwelltime_melt)
    plt.title('Dwell_Time_K-Means')
    pdf.savefig()
    plt.close()

    for s in range(k):
            # create average connectivity image
            X_conn=np.mean(X.iloc[np.where(P_hmm == s)[0]])
            visualize.plot_connectivity(X_conn)
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

pdf.close()
print('finished')


