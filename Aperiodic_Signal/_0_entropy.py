#!/usr/bin/env python

import matplotlib.backends.backend_pdf as pltpdf
from utils.visualize import plot_group_correlations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import antropy as ant
import argparse
import mne
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate different measures of enthropy.')
    parser.add_argument('input_dir', type=str, action='store',
                        help='folder name containing the data in .fdt and .set format')
    parser.add_argument('output_dir', type=str, action='store',
                        help='folder name where to save the power spectra')
    parser.add_argument('patient_information', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('--conditions', '-cond', nargs='*', action='store', default='Baseline Anesthesia',
                        help='The "task" or conditions you want to compare for example Baseline Anesthesia'
                             'can be only Base or Baseline and Anesthesia')
    args = parser.parse_args()

    # make ouput directory
    output_dir = os.path.join(args.output_dir, 'Entropy')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = pltpdf.PdfPages("{}/Entropy.pdf".format(output_dir))

    # load patient IDS
    info = pd.read_csv(args.patient_information, sep='\t')
    P_IDS = info['Patient']
    outcome = info['Outcome']
    group = info['Group']

    toplot = pd.DataFrame()
    toplot['ID'] = info['Patient']
    toplot['outcome'] = info['Outcome'].astype(int)
    try:
        toplot['scale'] = info['CRSR']
    except:
        toplot['scale'] = info['GCS_sedoff']
    toplot['Group'] = info['Group']
    toplot['Age'] = info['Age'].astype(int)

    for c in args.conditions:
        # define empty DataFrames to save
        Perm_ent = []
        Spect_ent = []
        Hjor_comp = []
        Hjor_mobi = []

        for p_id in P_IDS:
            """
            1)    IMPORT DATA
            """
            try:
                input_fname = args.input_dir + "/{}_{}.set".format(p_id, c)
                raw = mne.io.read_raw_eeglab(input_fname)
            except FileNotFoundError:
                input_fname = args.input_dir + "/{}_task-{}_eeg.set".format(p_id, c)
                raw = mne.io.read_raw_eeglab(input_fname)

            nr_channels = raw.info['nchan']
            raw_down = raw.copy().resample(sfreq=250)

            """
                2)    Compute Antropy and Complexity 
            """
            tmp_Perm_ent = []

            for i in range(nr_channels):
                x = raw_down[i][0][0]
                # Permutation entropy
                tmp_Perm_ent.append(ant.perm_entropy(x, normalize=True))

            Perm_ent.append(np.mean(tmp_Perm_ent))
            # Spectral entropy
            Spect_ent.append(np.mean(ant.spectral_entropy(raw_down[:][0], sf=100, method='welch', normalize=True)))
            # Hjorth mobility and complexity
            Hjor_mobi.append(np.mean(ant.hjorth_params(raw_down[:][0])[0]))
            Hjor_comp.append(np.mean(ant.hjorth_params(raw_down[:][0])[1]))
            print( "Calculation finished for  " + p_id)

        toplot['Perm_ent_{}'.format(c)] = Perm_ent
        toplot['Spect_ent_{}'.format(c)] = Spect_ent
        toplot['Hjor_mobi_{}'.format(c)] = Hjor_comp
        toplot['Hjor_comp_{}'.format(c)] = Hjor_comp

    if len(args.conditions) == 2:
        toplot['Perm_ent_diff'] = toplot['Perm_ent_{}'.format(args.conditions[0])] - toplot['Perm_ent_{}'.format(args.conditions[1])]
        toplot['Spect_ent_diff'] = toplot['Spect_ent_{}'.format(args.conditions[0])] - toplot['Spect_ent_{}'.format(args.conditions[1])]
        toplot['Hjor_mobi_diff'] = toplot['Hjor_mobi_{}'.format(args.conditions[0])] - toplot['Hjor_mobi_{}'.format(args.conditions[1])]
        toplot['Hjor_comp_diff'] = toplot['Hjor_comp_{}'.format(args.conditions[0])] - toplot['Hjor_comp_{}'.format(args.conditions[1])]



    if len(np.unique(toplot['Group'])) > 1:
        plot_group_correlations(data=toplot, start=5, category='outcome', group=group, pdf=pdf)

    if len(args.conditions) == 2:
        plt.figure()
        for i in range(len(P_IDS)):
            if toplot['outcome'][i] == 0:
                col = 'r'
            if toplot['outcome'][i] == 1:
                col = 'g'
            plt.plot([1, 2], [toplot['Perm_ent_{}'.format(args.conditions[0])][i],
                              toplot['Perm_ent_{}'.format(args.conditions[1])][i]], col)
        plt.xticks([1, 2], [args.conditions[0], args.conditions[1]])
        plt.title("Permutation_entropy")
        pdf.savefig()
        plt.close()

    pdf.close()

