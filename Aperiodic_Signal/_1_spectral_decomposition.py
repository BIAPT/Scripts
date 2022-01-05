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
                        help='folder name containing the data in epoched .fif data')
    parser.add_argument('output_dir', type=str, action='store',
                        help='folder name where to save the power spectra')
    parser.add_argument('patient_information', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('condition', type=str, action='store',
                        help='The "task" or condition you want to analyze')
    args = parser.parse_args()

    """
           1)    PREPARE IN-AND OUTPUT
    """

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
    # 2) Multitaper power density
    power_spectra_mt = list()

    for p_id in P_IDS:
        """
        2)      IMPORT DATA
        """
        # define epoch name
        input_fname = args.input_dir + "/sub-{}/eeg/epochs_{}_{}.fif".format(p_id, p_id, args.condition)
        raw_epochs = mne.read_epochs(input_fname)

        # remove channels marked as bad and non-brain channels
        raw_epochs.drop_channels(raw_epochs.info['bads'])

        #crop data if necessary
        raw_len = len(raw_epochs)

        if raw_len > 30:
            # choose last 30 epochs
            epochs_cropped = raw_epochs[raw_len-30:raw_len]
        else:
            epochs_cropped = raw_epochs.copy()

        """
            3)    Plot PSD for all electrodes 
        """
        # show a general PSD
        fig, ax = plt.subplots(nrows=1, ncols=1)
        epochs_cropped.plot_psd(ax=ax, fmin=0.5, fmax=50, dB=True, show=False,average=False)
        ax.set_title(p_id + "  " + "PSD")
        pdf.savefig(fig)
        plt.close()

        """
            4)    Compute PSD with Welch Method 
        """
        # 1) using the Welch Method
        # Hanning window of 2 s
        # 1 s overlap
        psds_welch, freqs_welch = psd_welch(epochs_cropped, fmin= 0.5, fmax=50, n_fft = 2*250, n_per_seg = 2*250, n_overlap = 1*250)
        psds_welch_db = 10 * np.log10(psds_welch) # convert to dB

        # save the PSD for later analysis
        power_spectra_welch.append(psds_welch)

        """
            5)    Compute PSD with Multitaper Method 
        """

        # 1) using the Multitaper Method
        # bandwith (frequency smoothing) of 0.5
        # resulting number of tapers: 4
        # See this paper: https://journals.physiology.org/doi/full/10.1152/physiol.00062.2015

        psds_mt, freqs_mt = psd_multitaper(epochs_cropped, fmin=0.5, fmax=50, bandwidth=0.5)
        psds_mt_db = 10 * np.log10(psds_mt) # convert to dB

        # save the PSD for later analysis
        power_spectra_mt.append(psds_mt)

        """
            6)    Plot PSD Welch and Multitaper 
        """
        # plot results
        fig = plot_two_curves(x1=freqs_welch, x2=freqs_mt,
                              y1=psds_welch_db.mean(0), y2=psds_mt_db.mean(0),
                              c1='green', c2='orange',
                              l1='Welch', l2='Multitaper',
                              title='{} Power spectral density'.format(p_id),
                              lx='Frequency (Hz)', ly='Power Spectral Density (dB)'
                              )
        pdf.savefig(fig)
        plt.close(fig)
        plt.show()

    # Add IDs at the end of the list
    power_spectra_mt.append(P_IDS)
    power_spectra_welch.append(P_IDS)

    with open('{}/PSD_Welch_{}.pkl'.format(output_dir,args.condition), 'wb') as f:
        pickle.dump(power_spectra_welch, f)

    with open('{}/PSD_Multitaper_{}.pkl'.format(output_dir,args.condition), 'wb') as f:
        pickle.dump(power_spectra_mt, f)

    freqs_mt = pd.DataFrame(freqs_mt)
    freqs_welch = pd.DataFrame(freqs_welch)
    freqs_welch.to_csv('{}\Frequency_Welch_{}.txt'.format(output_dir, args.condition), index = None, header=None, sep=' ')
    freqs_mt.to_csv('{}\Frequency_Multitaper_{}.txt'.format(output_dir, args.condition), index = None, header=None,sep=' ')

    pdf.close()

