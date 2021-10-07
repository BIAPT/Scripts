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

from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model

INPUT_DIR = "data_WSAS/"
pdf = matplotlib.backends.backend_pdf.PdfPages("FOOOF_WSAS_30-45.pdf")

info = pd.read_table("data_WSAS/WSAS_information.txt")
P_IDS = info['Patient']

offset_Base = []
offset_Anes = []
exponent_Base = []
exponent_Anes = []
all_psds_Base = []
all_psds_Anes = []

for p_id in P_IDS:
    """
    1)    IMPORT DATA
    """
    input_fname_B = INPUT_DIR + "{}_Base_5min.set".format(p_id)
    input_fname_A = INPUT_DIR + "{}_Anes_5min.set".format(p_id)

    raw_A = mne.io.read_raw_eeglab(input_fname_A)
    raw_B = mne.io.read_raw_eeglab(input_fname_B)

    # Construct Epochs
    epochs_A = mne.make_fixed_length_epochs(raw_A, duration=10.0, preload=False, reject_by_annotation=False, proj=True, overlap=0.0,
                                 verbose=None)
    epochs_B = mne.make_fixed_length_epochs(raw_B, duration=10.0, preload=False, reject_by_annotation=False, proj=True, overlap=0.0,
                                 verbose=None)

    """
        2)    Compute and plot PSD
    """
    psds_A, freqs = psd_multitaper(epochs_A, fmin=0.5, fmax=45, n_jobs=1, )
    psds_B, freqs = psd_multitaper(epochs_B, fmin=0.5, fmax=45, n_jobs=1, )

    psds_A_mean = psds_A.mean(0).mean(0)
    psds_B_mean = psds_B.mean(0).mean(0)

    psds_A_log = np.log10(psds_A)
    psds_B_log = np.log10(psds_B)

    psds_A_log_mean = psds_A_log.mean(0).mean(0)
    psds_B_log_mean = psds_B_log.mean(0).mean(0)

    psds_A_log_std = psds_A_log.mean(0).std(0)
    psds_B_log_std = psds_B_log.mean(0).std(0)

    # PLot unlog frequency
    f, ax = plt.subplots()
    ax.plot(freqs, psds_A_log_mean, color='b')
    ax.fill_between(freqs, psds_A_log_mean - psds_A_log_std, psds_A_log_mean + psds_A_log_std,
                    color='b', alpha=.5)
    ax.plot(freqs, psds_B_log_mean, color='r')
    ax.fill_between(freqs, psds_B_log_mean - psds_B_log_std, psds_B_log_mean + psds_B_log_std,
                    color='r', alpha=.5)
    ax.set(title='{} Multitaper PSD (gradiometers)'.format(p_id), xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')
    pdf.savefig()
    plt.close()

    # PLot log frequency
    f, ax = plt.subplots()
    ax.plot(np.log10(freqs), psds_A_log_mean, color='b')
    ax.fill_between(np.log10(freqs), psds_A_log_mean - psds_A_log_std, psds_A_log_mean + psds_A_log_std,
                    color='b', alpha=.5)
    ax.plot(np.log10(freqs), psds_B_log_mean, color='r')
    ax.fill_between(np.log10(freqs), psds_B_log_mean - psds_B_log_std, psds_B_log_mean + psds_B_log_std,
                    color='r', alpha=.5)
    ax.set(title='{} Multitaper PSD (gradiometers)'.format(p_id), xlabel='LOG(Frequency) (Hz)',
           ylabel='Power Spectral Density (dB)')
    pdf.savefig()
    plt.close()
    plt.show()


    all_psds_Anes.append(psds_A_log_mean)
    all_psds_Base.append(psds_B_log_mean)

    """
        3)    Compute nonperiodic signal
    """
    # Set the frequency range to fit the model
    freq_range = [30, 45]

    # Initialize power spectrum model objects and fit the power spectra
    fm_A = FOOOF(aperiodic_mode='fixed')
    fm_A.fit(freqs, psds_A_mean,freq_range=freq_range)

    fm_B = FOOOF(aperiodic_mode='fixed')
    fm_B.fit(freqs, psds_B_mean,freq_range=freq_range)

    # Aperiodic parameters
    offset_Anes.append(fm_A.aperiodic_params_[0])
    offset_Base.append(fm_B.aperiodic_params_[0])

    exponent_Anes.append(fm_A.aperiodic_params_[1])
    exponent_Base.append(fm_B.aperiodic_params_[1])

    # Plot an example power spectrum, with a model fit
    #fm_A.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'})
    #fm_B.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'})
    #plt.show()
    #pdf.savefig()
    #plt.close()

"""
Plot group level results: 
"""
toplot = pd.DataFrame()
toplot['ID'] = P_IDS
toplot['outcome'] = info['Outcome']
toplot['CRSR'] = info['CRSR'].astype(int)
toplot['Age'] = info['Age'].astype(int)
#mean
toplot['Offset_Base'] = offset_Base
toplot['Offset_Anes'] = offset_Anes
toplot['Offset_Diff'] = toplot['Offset_Base']-toplot['Offset_Anes']
toplot['Exponent_Base'] = exponent_Base
toplot['Exponent_Anes'] = exponent_Anes
toplot['Exponent_Diff'] = toplot['Exponent_Base']-toplot['Exponent_Anes']


# 0 = Non-recovered
# 1 = Recovered

for i in toplot.columns[4:]:
    plt.figure()
    sns.boxplot(x='outcome', y = i, data=toplot)
    sns.stripplot(x='outcome', y = i, size=4, color=".3", data=toplot)
    plt.xticks([0, 1], ['Nonreco', 'Reco'])
    plt.title(i)
    pdf.savefig()
    plt.close()

    # plot CRSR-Offset
    fig = plt.figure()
    corr = pearsonr(toplot["CRSR"], toplot[i])
    sns.regplot(x='CRSR', y=i, data=toplot)
    plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


plt.figure()
for i in range(len(P_IDS)):
    if toplot['outcome'][i] == 0:
        c = '--g'
    if toplot['outcome'][i] == 1:
        c = 'r'
    plt.plot([1, 2], [toplot['Exponent_Base'][i], toplot['Exponent_Anes'][i]],c)
plt.xticks([1,2],['Baseline','Anesthesia'])
plt.title("Exponent")
pdf.savefig()
plt.close()

plt.figure()
for i in range(len(P_IDS)):
    if toplot['outcome'][i] == 0:
        c = '--g'
    if toplot['outcome'][i] == 1:
        c = 'r'
    plt.plot([1, 2], [toplot['Offset_Base'][i], toplot['Offset_Anes'][i]],c)
plt.xticks([1,2],['Baseline','Anesthesia'])
plt.title("Offset")
pdf.savefig()
plt.close()



pdf.close()

