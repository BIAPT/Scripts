#!/usr/bin/env python

from mne.time_frequency import psd_multitaper, psd_welch
import matplotlib.backends.backend_pdf as pltpdf
from utils.visualize import plot_two_curves
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pickle
import mne
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate the spectral density.')
    parser.add_argument('input_dir', type=str, action='store',
                        help='folder name containing the data in .fdt and .set format')
    parser.add_argument('output_dir', type=str, action='store',
                        help='folder name where to save the power spectra')
    parser.add_argument('patient_information', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('--condition', '-cond', type=str, action='store',
                        help='The "task" or condition you want to analyze')
    args = parser.parse_args()

    # make ouput directory
    output_dir = os.path.join(args.output_dir, args.condition)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = pltpdf.PdfPages("{}/spectral_decomposition_{}.pdf".format(output_dir, args.condition))

    # load patient IDS
    info = pd.read_csv(args.patient_information, sep='\t')
    P_IDS = info['Patient']

    # define empty DataFrames to save
    # 1) Welch Power density
    power_spectra_welch = list()
    power_spectra_welch_avg = []
    # 2) Multitaper power density
    power_spectra_mt = list()
    power_spectra_mt_avg = []

    for p_id in P_IDS:
        """
        1)    IMPORT DATA
        """
        input_fname = args.input_dir + "/sub-{}/eeg/epochs_{}_{}.fif".format(p_id, p_id, args.condition)
        raw_epochs = mne.read_epochs(input_fname)

        #try:
            #input_fname = args.input_dir + "/{}_{}.set".format(p_id, args.condition)
            #raw = mne.io.read_raw_eeglab(input_fname)
        #except FileNotFoundError:
        #    input_fname = args.input_dir + "/{}_task-{}_eeg.set".format(p_id, args.condition)
        #    raw = mne.io.read_raw_eeglab(input_fname)

        #downsample data
        #raw_downsampled = raw.copy().resample(sfreq=250)

        # show the duration of your signal (number of epochs)
        #raw_len = raw_epochs.times[-1]
        raw_len = len(raw_epochs)

        #crop data if necessary
        if raw_len > 5 * 60:
            if args.condition == 'Base_5min':
                # choose first 5 min
                raw_cropped = raw_downsampled.copy().crop(tmax=(5*60))
            if args.condition == 'Anes_5min':
                # choose last 5 min
                raw_cropped = raw_downsampled.copy().crop(tmin=raw_len-(5*60))
            if args.condition == 'Reco_5min':
                # choose last 5 min
                raw_cropped = raw_downsampled.copy().crop(tmin=raw_len-(5*60))
            if args.condition == "sedon1":
                # choose first 5 min
                raw_cropped = raw_downsampled.copy().crop(tmax=(5*60))
            if args.condition == "sedoff":
                # choose last 5 min
                raw_cropped = raw_downsampled.copy().crop(tmin=raw_len-(5*60))
            if args.condition == "sedon2":
                # choose last 5 min
                raw_cropped = raw_downsampled.copy().crop(tmin=raw_len-(5*60))
            else:
                raw_cropped = raw_downsampled.copy()
        else:
            raw_cropped = raw_downsampled.copy()

        # Construct Epochs
        epochs = mne.make_fixed_length_epochs(raw_cropped, duration=10.0, preload=False, reject_by_annotation=False, proj=True,
                                              overlap=0.0, verbose=None)

        """
            2)    Compute and plot PSD 
        """
        # 1) using the Welch Method
        psds_welch, freqs_welch = psd_welch(epochs, fmin=0.1, fmax=50)
        # average over channels and time and log
        psds_log_welch = np.log10(psds_welch).mean(0)
        power_spectra_welch_avg.append(psds_welch.mean(0).mean(0))
        power_spectra_welch.append(psds_welch)

        # 1) using the Multitaper Method
        psds_mt, freqs_mt = psd_multitaper(epochs, fmin=0.1, fmax=50)
        # average over channels and time and log
        psds_log_mt = np.log10(psds_mt).mean(0)
        power_spectra_mt_avg.append(psds_mt.mean(0).mean(0))
        power_spectra_mt.append(psds_mt)

        # plot results
        fig = plot_two_curves(x1=freqs_welch, x2=freqs_mt,
                              y1=psds_log_welch, y2=psds_log_mt,
                              c1='green', c2='orange',
                              l1='Welch', l2='Multitaper',
                              title='{} Power spectral density'.format(p_id),
                              lx='Frequency (Hz)', ly='Power Spectral Density (dB)')
        pdf.savefig(fig)
        plt.close(fig)

    # Add IDs at the end of the list
    power_spectra_mt.append(P_IDS)
    power_spectra_welch.append(P_IDS)

    with open('{}/PSD_Welch_{}.pkl'.format(output_dir,args.condition), 'wb') as f:
        pickle.dump(power_spectra_welch, f)

    with open('{}/PSD_Multitaper_{}.pkl'.format(output_dir,args.condition), 'wb') as f:
        pickle.dump(power_spectra_mt, f)

    power_spectra_mt_avg = pd.DataFrame(power_spectra_mt_avg)
    power_spectra_mt_avg.insert(0, 'ID', P_IDS)
    power_spectra_welch_avg = pd.DataFrame(power_spectra_welch_avg)
    power_spectra_welch_avg.insert(0, 'ID', P_IDS)
    freqs_mt = pd.DataFrame(freqs_mt)
    freqs_welch = pd.DataFrame(freqs_welch)

    power_spectra_welch_avg.to_csv('{}\Power_spectra_Welch_{}.txt'.format(output_dir, args.condition), index = None, sep=' ')
    freqs_welch.to_csv('{}\Frequency_Welch_{}.txt'.format(output_dir, args.condition), index = None, header=None, sep=' ')
    power_spectra_mt_avg.to_csv('{}\Power_spectra_Multitaper_{}.txt'.format(output_dir, args.condition), index = None, sep=' ')
    freqs_mt.to_csv('{}\Frequency_Multitaper_{}.txt'.format(output_dir, args.condition), index = None, header=None,sep=' ')

    pdf.close()

