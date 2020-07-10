import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
import matplotlib.backends.backend_pdf
from statsmodels.graphics import tsaplots

#part = ['13', '18', '05', '11', '19', '02', '20', '22', '12', '10', '09']
part = ['09']

for p in part:
    pdf = matplotlib.backends.backend_pdf.PdfPages("Autocorrelation_dPLI_Part_{}.pdf".format(p))

    connectivity=['dPLI']
    #connectivity=['wPLI','dPLI']

    for c in connectivity:
        data = pd.read_pickle('../data/WholeBrain_{}_10_1_alpha.pickle'.format(c))
        areas=data.columns[4:]

        data_p=data.query("ID == '{}'".format(p))
        data_p_Base=data_p.query("Phase == 'Base'")
        data_p_Anes=data_p.query("Phase == 'Anes'")
        #data_p_Reco=data_p.query("Phase == 'Reco'")

        for i in range(len(areas)):

            fig = plt.figure(figsize=(17, 8))
            fig.suptitle('Part: {} Left autocorrelation'.format(p), size=16)
            plt.subplot(121)
            ax1=plt.subplot(1,2,1)
            tsaplots.plot_acf(data_p_Base[areas[i]], lags=100,ax=ax1)
            ax1.set_title('Baseline  ' + areas[i])
            ax2=plt.subplot(1,2,2)
            tsaplots.plot_acf(data_p_Anes[areas[i]], lags=100,ax=ax2)
            ax2.set_title('Anesthesia' + areas[i])

            pdf.savefig(fig)
            plt.close()

    pdf.close()


