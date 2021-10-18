#!/usr/bin/env python

from mne.time_frequency import psd_multitaper, psd_welch
from fooof.plts.annotate import plot_annotated_model
from sklearn.linear_model import LinearRegression
import matplotlib.backends.backend_pdf as pltpdf
from fooof.sim.gen import gen_power_spectrum
from fooof.plts.spectra import plot_spectrum
from fooof.sim.utils import set_random_seed
from utils.visualize import plot_group_correlations
from utils.visualize import plot_two_curves
from utils.visualize import plot_cat_curves
from utils.visualize import plot_correlation
import matplotlib.pyplot as plt
from fooof import FOOOF
import pandas as pd
import numpy as np
import pickle
import argparse
import mne
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate the aperiodic signal portion using different Models.')
    parser.add_argument('data_dir', type=str, action='store',
                        help='folder name containing the data in .fdt and .set format')
    parser.add_argument('input_dir', type=str, action='store',
                        help='folder name containing the PSD data')
    parser.add_argument('patient_information', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('--conditions', '-cond', nargs='*', action='store', default='Baseline Anesthesia',
                        help='The "task" or conditions you want to compare for example Baseline Anesthesia'
                             'can be only Base or Baseline and Anesthesia')
    parser.add_argument('--frequency_range', '-freq', nargs='*', action='store', default='1 40',
                        help='The freqency band to calculate the aperiodic signal on. For example 1 20')
    parser.add_argument('--method', nargs=1, action='store', default='Multitaper', choices=('Multitaper','Welch'),
                        help='The method used for Spectral decomposition in Step 1')
    args = parser.parse_args()

    nr_cond = len(args.conditions)
    frequency_range = [int(args.frequency_range[0]), int(args.frequency_range[1])]

    # make ouput directory
    output_dir = os.path.join(args.input_dir, 'topoplot_aperiodic')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = pltpdf.PdfPages("{}/topoplot_aperiodic_signal_{}_{}.pdf".format(output_dir, frequency_range[0],frequency_range[1]))

    # load patient info
    info = pd.read_csv(args.patient_information,sep = '\t')
    P_IDS = info['Patient']
    outcome = info['Outcome']
    group = info['Group']

    # load conditions
    cond_B = str(args.conditions[0])
    if nr_cond == 2:
        cond_A = str(args.conditions[1])

    # load frequency information
    datapath = os.path.join(args.input_dir, cond_B, 'Frequency_{}_{}.txt'.format(args.method, cond_B))
    freqs_B = pd.read_csv(datapath, sep=' ', header=None)
    freqs_B = np.squeeze(freqs_B)

    if nr_cond == 2:
        datapath = os.path.join(args.input_dir, cond_A, 'Frequency_{}_{}.txt'.format(args.method, cond_A))
        freqs_A = pd.read_csv(datapath, sep=' ', header=None)
        freqs_A = np.squeeze(freqs_A)


    # load power spectral data
    PSD_B = pickle.load(open("{}/{}/PSD_{}_{}.pkl".format(args.input_dir, cond_B, args.method, cond_B), "rb"))
    PSD_B_ID = PSD_B[-1]

    if nr_cond == 2:
        PSD_A = pickle.load(open("{}/{}/PSD_{}_{}.pkl".format(args.input_dir, cond_A, args.method, cond_A), "rb"))
        PSD_A_ID = PSD_A[-1]

    for p_id in P_IDS:

        # imput raw data (needed later for plotting only)
        input_fname = "{}/{}_{}.set".format(args.data_dir, p_id, cond_B)
        raw_B = mne.io.read_raw_eeglab(input_fname)
        if nr_cond == 2:
            input_fname = "{}/{}_{}.set".format(args.data_dir, p_id, cond_A)
            raw_A = mne.io.read_raw_eeglab(input_fname)
            # compare electrodes to get overlying ones
            ch_A = np.array(raw_A.info.ch_names)
            ch_B = np.array(raw_B.info.ch_names)
            keep_A = np.isin(ch_A,ch_B)
            keep_B = np.isin(ch_B,ch_A)

        # select individual PSD values, depending on ID
        index_id = np.where(PSD_B_ID == p_id)[0][0]
        psd_B_p = PSD_B[index_id]
        if nr_cond == 2:
            index_id = np.where(PSD_A_ID == p_id)[0][0]
            psd_A_p = PSD_A[index_id]

        # only use defined frequency band
        index_toselect = np.where((freqs_B >= frequency_range[0]) & (freqs_B <= frequency_range[1]))[0]

        freqs_select = freqs_B[index_toselect]
        #   Log frequency for Linear Regression
        freqs_select_log = np.array(np.log10(freqs_select))

        if nr_cond == 1:
            psd_B_p_select = psd_B_p[:, :, index_toselect]
            psd_B_p_select_avg = np.mean(psd_B_p_select, axis=0)

        if nr_cond == 2:
            psd_B_p_select = psd_B_p[:, keep_B, :]
            psd_B_p_select = psd_B_p_select[:, :, index_toselect]
            psd_B_p_select_avg = np.mean(psd_B_p_select, axis=0)

            psd_A_p_select = psd_A_p[:, keep_A, :]
            psd_A_p_select = psd_A_p_select[:, :, index_toselect]
            psd_A_p_select_avg = np.mean(psd_A_p_select, axis=0)

        # initiate empty frame for electrodes:
        nr_el_B = psd_B_p_select_avg.shape[0]
        exponent_B_fooof = np.empty(nr_el_B)
        exponent_B_lin = np.empty(nr_el_B)

        for i in range(nr_el_B):
            # FOOOF fit
            #fm_B = FOOOF(aperiodic_mode='fixed')
            #fm_B.fit(np.array(freqs_select), psd_B_p_select_avg[i])
            #exponent_B_fooof[i] = fm_B.aperiodic_params_[1]
            # Linear regression fit
            lm_B = LinearRegression()
            lm_B.fit(freqs_select_log.reshape(-1, 1), np.log10(psd_B_p_select_avg[i]))
            exponent_B_lin[i]=lm_B.coef_[0]

        exponent_B_fooof = list(exponent_B_fooof)
        exponent_B_lin = list(exponent_B_lin)

        if nr_cond == 2:
            nr_el_A = psd_A_p_select_avg.shape[0]
            exponent_A_fooof = np.empty(nr_el_A)
            exponent_A_lin = np.empty(nr_el_A)

            for i in range(nr_el_A):
                # FOOOF fit
                #fm_A = FOOOF(aperiodic_mode='fixed')
                #fm_A.fit(np.array(freqs_select), psd_A_p_select_avg[i])
                #exponent_A_fooof[i] = fm_A.aperiodic_params_[1]
                # Linear regression fit
                lm_A = LinearRegression()
                lm_A.fit(freqs_select_log.reshape(-1, 1), np.log10(psd_A_p_select_avg[i]))
                exponent_A_lin[i] = lm_A.coef_[0]

            exponent_A_fooof = list(exponent_A_fooof)
            exponent_A_lin = list(exponent_A_lin)


        if nr_cond == 2:
            raw_B.drop_channels(list(ch_B[np.invert(keep_B)]))
            raw_A.drop_channels(list(ch_A[np.invert(keep_A)]))

        fig = plt.figure()
        im, _ = mne.viz.plot_topomap(exponent_B_lin, raw_B.info, vmin=-4, vmax=2, show=False)
        fig.colorbar(im)
        fig.subplots_adjust(top=0.8)
        fig.suptitle(p_id+"__"+cond_B)
        pdf.savefig(fig)
        plt.close()

        if nr_cond == 2:
            fig = plt.figure()
            im, _ = mne.viz.plot_topomap(exponent_A_lin, raw_A.info, vmin= -4, vmax=2, show=False)
            fig.colorbar(im)
            fig.subplots_adjust(top=0.8)
            fig.suptitle(p_id + "__" + cond_A)
            pdf.savefig(fig)
            plt.close()

            # calculate the difference
            #exponent_diff_fooof = list(np.array(exponent_B_fooof)-np.array(exponent_A_fooof))
            exponent_diff_lin = list(np.array(exponent_B_lin)-np.array(exponent_A_lin))

            fig = plt.figure()
            im, _ = mne.viz.plot_topomap(exponent_diff_lin, raw_A.info, vmin=-2, vmax=2, show=False)
            fig.colorbar(im)
            fig.subplots_adjust(top=0.8)
            fig.suptitle(p_id + "__" + 'Base - Anes ')
            pdf.savefig(fig)
            plt.close()

    pdf.close()

