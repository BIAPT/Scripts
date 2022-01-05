#!/usr/bin/env python

import matplotlib.backends.backend_pdf as pltpdf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    parser.add_argument('--model', '-model', nargs=1, action='store', default=['fooof'],
                        help='Can be the fooof model or linreg',choices=('fooof','linreg'))


    args = parser.parse_args()
    nr_cond = len(args.conditions)
    frequency_range = [int(args.frequency_range[0]), int(args.frequency_range[1])]
    model = args.model[0]
    input_dir = args.input_dir

    # make ouput directory
    output_dir = os.path.join(input_dir, 'aperiodic_signal')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = pltpdf.PdfPages("{}/topoplot_aperiodic_signal_{}_{}_{}.pdf".format
                          (output_dir, frequency_range[0],frequency_range[1],model))

    # load patient info
    info = pd.read_csv(args.patient_information,sep = '\t')
    P_IDS = info['Patient']
    outcome = info['Outcome']
    group = info['Group']

    # load conditions
    cond_B = str(args.conditions[0])
    if nr_cond == 2:
        cond_A = str(args.conditions[1])

    """
        1) load aperiodic data
    """
    params_B = pd.read_csv('{}\Params_space_{}_{}_{}_{}_{}.txt'.format
                           (output_dir, cond_B, frequency_range[0], frequency_range[1], model, 'all'),
                           sep=' ')

    if nr_cond == 2:
        params_A = pd.read_csv('{}\Params_space_{}_{}_{}_{}_{}.txt'.format
                               (output_dir, cond_A, frequency_range[0], frequency_range[1], model, 'all'),
                               sep=' ')

    for p_id in P_IDS:
        # select p_id from aperiodic dataframe
        params_B_id = params_B[params_B['ID']==p_id]
        # select only exponent data
        params_B_id = params_B_id['exponent'].values[0]
        # remove the brakets from the string (some artifact from the saving format)
        params_B_id = params_B_id.replace('[', '')
        params_B_id = params_B_id.replace(']', '')
        exponent_B_id = params_B_id.split()
        exponent_B_id = np.array(exponent_B_id).astype(float)

        if nr_cond == 2:
            # select p_id from aperiodic dataframe
            params_A_id = params_A[params_A['ID'] == p_id]
            # select only exponent data
            params_A_id = params_A_id['exponent'].values[0]
            # remove the brakets from the string (some artifact from the saving format)
            params_A_id = params_A_id.replace('[', '')
            params_A_id = params_A_id.replace(']', '')
            exponent_A_id = params_A_id.split()
            exponent_A_id = np.array(exponent_A_id).astype(float)

        # imput raw data (needed later for plotting only)
        input_fname = "{}/sub-{}/eeg/epochs_{}_{}.fif".format(args.data_dir, p_id, p_id, cond_B)
        # remove channels marked as bad and non-brain channels
        raw_B = mne.read_epochs(input_fname)
        raw_B.drop_channels(raw_B.info['bads'])

        # select all channels to visualize
        exponent_id_select_B = list(exponent_B_id)

        if nr_cond == 2:
            #input_fname = "{}/{}_{}.set".format(args.data_dir, p_id, cond_A)
            input_fname = "{}/sub-{}/eeg/epochs_{}_{}.fif".format(args.data_dir, p_id, p_id, cond_A)
            raw_A = mne.read_epochs(input_fname)
            # remove channels marked as bad and non-brain channels
            raw_A.drop_channels(raw_A.info['bads'])

            # compare electrodes to get overlying ones
            ch_A = np.array(raw_A.info.ch_names)
            ch_B = np.array(raw_B.info.ch_names)
            keep_A = np.isin(ch_A,ch_B)
            keep_B = np.isin(ch_B,ch_A)

            # select only the aperiodic components for both conditions
            exponent_id_select_B = list(exponent_B_id[keep_B])
            exponent_id_select_A = list(exponent_A_id[keep_A])

            # and do the same thing for the channel information
            raw_B.drop_channels(list(ch_B[np.invert(keep_B)]))
            raw_A.drop_channels(list(ch_A[np.invert(keep_A)]))

        fig = plt.figure()
        im, _ = mne.viz.plot_topomap(exponent_id_select_B, raw_B.info, vmin=-4, vmax=2, show=False)
        fig.colorbar(im)
        fig.subplots_adjust(top=0.8)
        fig.suptitle(p_id+"__"+cond_B)
        pdf.savefig(fig)
        plt.close()

        if nr_cond == 2:
            fig = plt.figure()
            im, _ = mne.viz.plot_topomap(exponent_id_select_A, raw_A.info, vmin= -4, vmax=2, show=False)
            fig.colorbar(im)
            fig.subplots_adjust(top=0.8)
            fig.suptitle(p_id + "__" + cond_A)
            pdf.savefig(fig)
            plt.close()

            # calculate the difference
            exponent_diff_lin = list(np.array(exponent_id_select_B )-np.array(exponent_id_select_A))

            fig = plt.figure()
            im, _ = mne.viz.plot_topomap(exponent_diff_lin, raw_A.info, vmin=-2, vmax=2, show=False)
            fig.colorbar(im)
            fig.subplots_adjust(top=0.8)
            fig.suptitle(p_id + "__" + 'Base - Anes ')
            pdf.savefig(fig)
            plt.close()

    pdf.close()

