import mne
import fooof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import matplotlib.backends.backend_pdf
import seaborn as sns
from fooof import FOOOF
from scipy.stats import pearsonr
from fooof.bands import Bands
from fooof.utils import trim_spectrum
from fooof.bands import Bands
from fooof.sim.gen import gen_power_spectrum
from fooof.plts.spectra import plot_spectra_shading
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model

INPUT_DIR = "data/"
pdf = matplotlib.backends.backend_pdf.PdfPages("Power_all_patients.pdf")

info = pd.read_table("data/DOC_Cluster_information.txt")
P_IDS = info['Patient']

gamma = []
beta = []
alpha = []
theta = []
delta = []

# Define our bands of interest
bands = Bands({'gamma' : (30, 50),
               'beta' : (13, 30),
               'alpha' : (8, 13),
               'theta' : (4, 8),
               'delta' : (1, 4)})

def calc_avg_power(freqs, powers, freq_range):
    """Helper function to calculate average power in a band."""

    _, band_powers = trim_spectrum(freqs, powers, freq_range)
    avg_power = np.mean(band_powers)

    return avg_power


for p_id in P_IDS:
    """
    1)    IMPORT DATA
    """
    input_fname = INPUT_DIR + "{}_Base_5min.set".format(p_id)
    raw = mne.io.read_raw_eeglab(input_fname)

    # Construct Epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=10.0, preload=False, reject_by_annotation=False, proj=True, overlap=0.0,
                                 verbose=None)

    """
        2)    Compute and plot PSD
    """
    f, ax = plt.subplots()
    psds, freqs = psd_multitaper(epochs, fmin=0.5, fmax=45, n_jobs=1, )
    psds_mean = psds.mean(0).mean(0)

    # Calculate the amount of alpha power in the baseline power spectrum
    gamma.append(calc_avg_power(freqs, psds_mean, bands.gamma))
    beta.append(calc_avg_power(freqs, psds_mean, bands.beta))
    alpha.append(calc_avg_power(freqs, psds_mean, bands.alpha))
    theta.append(calc_avg_power(freqs, psds_mean, bands.theta))
    delta.append(calc_avg_power(freqs, psds_mean, bands.delta))

"""
Plot group level results: 
"""
toplot = pd.DataFrame()
toplot['ID'] = P_IDS
toplot['outcome'] = info['Outcome']
toplot['CRSR'] = info['CRSR']
toplot['Age'] = info['Age']
#mean
toplot['Gamma'] = gamma
toplot['Beta'] = beta
toplot['Alpha'] = alpha
toplot['Theta'] = theta
toplot['Delta'] = delta

# 0 = Non-recovered
# 1 = CMD
# 2 = Recovered
# 3 = Healthy

toplot_DOC = toplot.query("outcome != 3")
toplot_DOC['CRSR'] = toplot_DOC['CRSR'].astype(int)
toplot_DOC['Age'] = toplot_DOC['Age'].astype(int)


for i in toplot.columns[4:]:
    plt.figure()
    sns.boxplot(x='outcome', y = i, data=toplot)
    sns.stripplot(x='outcome', y = i, size=4, color=".3", data=toplot)
    plt.xticks([0, 1, 2, 3], ['Nonreco', 'CMD', 'Reco', 'Healthy'])
    plt.title(i)
    pdf.savefig()
    plt.close()

    fig = plt.figure()
    corr = pearsonr(toplot_DOC["CRSR"], toplot_DOC[i])
    sns.regplot(x='CRSR', y=i, data=toplot_DOC)
    plt.title("r = " + str(corr[0]) + "\n p = " + str(corr[1]))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

pdf.close()

