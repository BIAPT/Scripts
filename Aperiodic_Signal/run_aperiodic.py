#!/usr/bin/env python

import mne
import fooof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.linear_model import LinearRegression
from fooof import FOOOF
from scipy.stats import pearsonr

from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model

if __name__ == '__main__':
    INPUT_DIR = "data/"
    pdf = matplotlib.backends.backend_pdf.PdfPages("results/FOOOF_and_LG_all_patients.pdf")

    info = pd.read_csv("data/DOC_Cluster_information.txt")
    P_IDS = info['Patient']

    offset = []
    exponent = []
    all_psds = []
    coeff = []
    intercept = []


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
        psds_log = np.log10(psds)
        psds_mean = psds.mean(0).mean(0)
        psds_log_mean = psds_log.mean(0).mean(0)
        psds_log_std = psds_log.mean(0).std(0)
        ax.plot(freqs, psds_log_mean, color='k')
        ax.fill_between(freqs, psds_log_mean - psds_log_std, psds_log_mean + psds_log_std,
                        color='k', alpha=.5)
        ax.set(title='{} Multitaper PSD (gradiometers)'.format(p_id), xlabel='Frequency (Hz)',
               ylabel='Power Spectral Density (dB)')
        pdf.savefig()
        plt.close()

        all_psds.append(psds_log_mean)

        """
            3)    Compute nonperiodic signal
        """
        # Set the frequency range to fit the model
        freq_range = [30, 45]

        fm = FOOOF(aperiodic_mode='fixed')
        fm.fit(freqs, psds_mean, freq_range)

        # Aperiodic parameters
        print('Aperiodic parameters: \n', fm.aperiodic_params_, '\n')
        offset.append(fm.aperiodic_params_[0])
        exponent.append(fm.aperiodic_params_[1])

        freqs_select = freqs[(freqs >= 30) & (freqs <= 45)]
        psds_log_mean = psds_log_mean[(freqs >= 30) & (freqs <= 45)]

        lm = LinearRegression()
        lm.fit(np.log10(freqs_select).reshape(-1, 1), psds_log_mean.reshape(-1, 1))

        coeff.append(lm.coef_[0][0])
        intercept.append(lm.intercept_[0])

    """
    Plot group level results: 
    """
    toplot = pd.DataFrame()
    toplot['ID'] = P_IDS
    toplot['outcome'] = info['Outcome']
    toplot['CRSR'] = info['CRSR']
    toplot['Age'] = info['Age']
    #mean
    toplot['Offset'] = offset
    toplot['Exponent'] = exponent
    toplot['Coefficient'] = coeff
    toplot['Intercept'] = intercept

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

        # plot CRSR-Offset
        fig = plt.figure()
        corr = pearsonr(toplot_DOC["CRSR"], toplot_DOC[i])
        sns.regplot(x='CRSR', y=i, data=toplot_DOC)
        plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    toplot['all_psds'] = all_psds

    psds_N_mean = np.mean(np.array(toplot.query("outcome == 0")['all_psds']))
    psds_C_mean = np.mean(np.array(toplot.query("outcome == 1")['all_psds']))
    psds_R_mean = np.mean(np.array(toplot.query("outcome == 2")['all_psds']))
    psds_H_mean = np.mean(np.array(toplot.query("outcome == 3")['all_psds']))

    psds_N_std = np.std(np.array(toplot.query("outcome == 0")['all_psds']))
    psds_C_std = np.std(np.array(toplot.query("outcome == 1")['all_psds']))
    psds_R_std = np.std(np.array(toplot.query("outcome == 2")['all_psds']))
    psds_H_std = np.std(np.array(toplot.query("outcome == 3")['all_psds']))

    f, ax = plt.subplots()
    ax.plot(freqs, psds_N_mean, color='k')
    ax.fill_between(freqs, psds_N_mean - psds_N_std, psds_N_mean + psds_N_std,
                    color='k', alpha=.5)
    ax.plot(freqs, psds_R_mean, color='r')
    ax.fill_between(freqs, psds_R_mean - psds_R_std, psds_R_mean + psds_R_std,
                    color='r', alpha=.5)
    ax.plot(freqs, psds_C_mean, color='g')
    ax.fill_between(freqs, psds_C_mean - psds_C_std, psds_C_mean + psds_C_std,
                    color='g', alpha=.5)
    ax.plot(freqs, psds_H_mean, color='b')
    ax.fill_between(freqs, psds_H_mean - psds_H_std, psds_H_mean + psds_H_std,
                    color='b', alpha=.5)
    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')
    pdf.savefig()

    f, ax = plt.subplots()
    ax.plot(np.log10(freqs), psds_N_mean, color='k')
    ax.fill_between(np.log10(freqs), psds_N_mean - psds_N_std, psds_N_mean + psds_N_std,
                    color='k', alpha=.5)
    ax.plot(np.log10(freqs), psds_R_mean, color='r')
    ax.fill_between(np.log10(freqs), psds_R_mean - psds_R_std, psds_R_mean + psds_R_std,
                    color='r', alpha=.5)
    ax.plot(np.log10(freqs), psds_C_mean, color='g')
    ax.fill_between(np.log10(freqs), psds_C_mean - psds_C_std, psds_C_mean + psds_C_std,
                    color='g', alpha=.5)
    ax.plot(np.log10(freqs), psds_H_mean, color='b')
    ax.fill_between(np.log10(freqs), psds_H_mean - psds_H_std, psds_H_mean + psds_H_std,
                    color='b', alpha=.5)
    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')
    pdf.savefig()

    f, ax = plt.subplots()
    for i in range(len(P_IDS)):
        if toplot['outcome'][i] == 0:
            c = 'k'
        if toplot['outcome'][i] == 1:
            c = 'g'
        if toplot['outcome'][i] == 2:
            c = 'r'
        if toplot['outcome'][i] == 3:
            c = 'b'
        ax.plot(freqs, toplot['all_psds'][i], color= c , alpha = .5)

    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')
    pdf.savefig()

    f, ax = plt.subplots()
    for i in range(len(P_IDS)):
        if toplot['outcome'][i] == 0:
            c = 'k'
        if toplot['outcome'][i] == 1:
            c = 'g'
        if toplot['outcome'][i] == 2:
            c = 'r'
        if toplot['outcome'][i] == 3:
            c = 'b'
        ax.plot(np.log10(freqs), toplot['all_psds'][i], color= c , alpha = .5)

    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')
    pdf.savefig()

    plt.close()



    pdf.close()

