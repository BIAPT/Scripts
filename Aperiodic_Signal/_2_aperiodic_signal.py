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
import argparse
import pickle
import mne
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate the aperiodic signal portion using different Models.')
    parser.add_argument('data_dir', type=str, action='store',
                        help='folder name containing the data in .fdt and .set format')
    parser.add_argument('input_dir', type=str, action='store',
                        help='folder name containing the data in .fdt and .set format')
    parser.add_argument('patient_information', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('--conditions', '-cond', nargs='*', action='store', default='Baseline Anesthesia',
                        help='The "task" or conditions you want to compare for example Baseline Anesthesia'
                             'can be only Base or Baseline and Anesthesia')
    parser.add_argument('--frequency_range', '-freq', nargs='*', action='store', default='1 40',
                        help='The freqency band to calculate the aperiodic signal on. For example 1 20')
    parser.add_argument('--method', nargs=1, action='store', default='Multitaper', choices=('Multitaper','Welch'),
                        help='The method used for Spectral decomposition in Step 1')
    parser.add_argument('--electrode', '-el', nargs=1, action='store', default=['all'],
                        help='On which electrode should the alperiodic slope be computed '
                             'default is all)')
    args = parser.parse_args()
    electrode = args.electrode

    nr_cond = len(args.conditions)
    frequency_range = [int(args.frequency_range[0]), int(args.frequency_range[1])]

    # make ouput directory
    output_dir = os.path.join(args.input_dir, 'aperiodic_signal')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = pltpdf.PdfPages("{}/aperiodic_signal_{}_{}_{}.pdf".format(output_dir, frequency_range[0],
                                                                    frequency_range[1], electrode))

    # load patient info
    info = pd.read_csv(args.patient_information,sep = '\t')
    P_IDS = info['Patient']
    outcome = info['Outcome']
    group = info['Group']

    # define empty DataFrames to save Baseline data
    offset_Base_fooof = []
    exponent_Base_fooof = []
    offset_Base_lin = []
    exponent_Base_lin = []
    missing_ID_B = []
    missing_ID_A = []

    # define empty DataFrames to save Anesthesia data
    if nr_cond == 2:
        offset_Anes_fooof = []
        exponent_Anes_fooof = []
        offset_Anes_lin = []
        exponent_Anes_lin = []

    cond_B = str(args.conditions[0])
    if nr_cond == 2:
        cond_A = str(args.conditions[1])

    # load power spectral data
    PSD_B = pickle.load(open("{}/{}/PSD_{}_{}.pkl".format(args.input_dir, cond_B, args.method, cond_B), "rb"))
    PSD_B_ID = PSD_B[-1]

    if nr_cond == 2:
        PSD_A = pickle.load(open("{}/{}/PSD_{}_{}.pkl".format(args.input_dir, cond_A, args.method, cond_A), "rb"))
        PSD_A_ID = PSD_A[-1]

    # prepare empty dataframe for PSD per patient and selected average
    psd_B = []
    if nr_cond == 2:
        psd_A = []

    for p_id in P_IDS:
        # select individual PSD values, depending on ID
        index_id = np.where(PSD_B_ID == p_id)[0][0]
        psd_B_p = PSD_B[index_id]
        if nr_cond == 2:
            index_id = np.where(PSD_A_ID == p_id)[0][0]
            psd_A_p = PSD_A[index_id]

        if  electrode[0] == 'all':
            psd_B.append(psd_B_p.mean(0).mean(0))
            if nr_cond == 2:
                psd_A.append(psd_A_p.mean(0).mean(0))

        else:
            # imput raw data (needed later for selection of the electrode)
            input_fname = "{}/{}_{}.set".format(args.data_dir, p_id, cond_B)
            raw_B = mne.io.read_raw_eeglab(input_fname)

            # If this electrode is in the given EEG set:
            if np.isin(np.array(raw_B.info.ch_names),electrode).any():
                select_ch = np.where(np.array(raw_B.info.ch_names) == electrode)[0][0]
                # average over time
                psd_B_p = psd_B_p.mean(0)
                psd_B.append(psd_B_p[select_ch,:])
            else:
                missing_ID_B.append(p_id)

            if nr_cond == 2:
                # imput raw data (needed later for selection of the electrode)
                input_fname = "{}/{}_{}.set".format(args.data_dir, p_id, cond_A)
                raw_A = mne.io.read_raw_eeglab(input_fname)

                # If this electrode is in the given EEG set:
                if np.isin(np.array(raw_A.info.ch_names), electrode).any():
                    select_ch = np.where(np.array(raw_A.info.ch_names) == electrode)[0][0]
                    # average over time
                    psd_A_p = psd_A_p.mean(0)
                    psd_A.append(psd_A_p[select_ch, :])
                else:
                    missing_ID_A.append(p_id)

    psd_B = pd.DataFrame(psd_B)
    IDS_B = P_IDS[np.invert(np.isin(P_IDS,missing_ID_B))]
    psd_B['ID'] = IDS_B.reset_index(drop = True)
    outcome_B = outcome[np.invert(np.isin(P_IDS,missing_ID_B))].reset_index(drop = True)
    group_B = group[np.invert(np.isin(P_IDS,missing_ID_B))].reset_index(drop = True)

    if nr_cond == 2:
        psd_A = pd.DataFrame(psd_A)
        IDS_A = P_IDS[np.invert(np.isin(P_IDS, missing_ID_A))]
        psd_A['ID'] = IDS_A.reset_index(drop=True)
        outcome_A = outcome[np.invert(np.isin(P_IDS,missing_ID_A))].reset_index(drop = True)
        group_A = group[np.invert(np.isin(P_IDS,missing_ID_A))].reset_index(drop = True)

    # load frequencies
    datapath = os.path.join(args.input_dir, cond_B, 'Frequency_{}_{}.txt'.format(args.method, cond_B))
    freqs_B = pd.read_csv(datapath, sep=' ', header=None)
    freqs_B = np.squeeze(freqs_B)

    if nr_cond == 2:
        datapath = os.path.join(args.input_dir, cond_A, 'Frequency_{}_{}.txt'.format(args.method, cond_A))
        freqs_A = pd.read_csv(datapath, sep=' ', header=None)
        freqs_A = np.squeeze(freqs_A)

    if nr_cond == 2:
        # plot both signals
        fig = plot_two_curves(x1=np.log10(freqs_B), x2=np.log10(freqs_A),
                              y1=np.log10(psd_B.drop(columns=['ID'])), y2=np.log10(psd_A.drop(columns=['ID'])),
                              c1='red', c2='blue',
                              l1='Wake', l2='Anesthesia',
                              title='Power spectral density',
                              lx='log Frequency (Hz)', ly='log Power Spectral Density (dB)')
        pdf.savefig(fig)
        plt.close(fig)

    # plot PSD according to outcome:

    fig = plot_cat_curves(np.log10(psd_B.drop(columns=['ID'])), freqs_B, outcome_B, group_B,
                    title='{} PSD_el_{}\nmissimg_{}'.format(args.conditions[0],electrode,missing_ID_B),
                          lx='Frequency', ly='Log(Power)')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig = plot_cat_curves(np.log10(psd_B.drop(columns=['ID'])), np.log10(freqs_B), outcome_B, group_B,
                    title='{} PSD'.format(args.conditions[0]), lx='Log(Frequency)', ly='Log(Power)')
    pdf.savefig(fig)
    plt.close(fig)

    if nr_cond == 2:
        fig = plot_cat_curves(np.log10(psd_A.drop(columns=['ID'])), freqs_A, outcome_A, group_A,
                              title='{} PSD'.format(args.conditions[1]), lx='Frequency', ly='Log(Power)')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_cat_curves(np.log10(psd_A.drop(columns=['ID'])), np.log10(freqs_A), outcome_A, group_A,
                              title='{} PSD'.format(args.conditions[1]), lx='Log(Frequency)', ly='Log(Power)')
        pdf.savefig(fig)
        plt.close(fig)

    """
    #   Calculate Aperiodic signal
    """
    for p_id in P_IDS[np.invert(np.isin(P_IDS,missing_ID_B))]:
        psds_B_id = psd_B.query("ID == '{}'".format(p_id))
        psds_B_id = psds_B_id.drop(columns=['ID'])

        if nr_cond == 2:
            psds_A_id = psd_A.query("ID == '{}'".format(p_id))
            psds_A_id = psds_A_id.drop(columns=['ID'])

        # only use defined frequency band
        index_toselect = np.where((freqs_B >= frequency_range[0]) & (freqs_B <= frequency_range[1]))[0]

        freqs_select = freqs_B[index_toselect]
        psds_B_id_select = psds_B_id.iloc[:,index_toselect]
        psds_B_id_select = np.squeeze(np.array(psds_B_id_select))

        if nr_cond == 2:
            psds_A_id_select = psds_A_id.iloc[:, index_toselect]
            psds_A_id_select = np.squeeze(np.array(psds_A_id_select))

        # Initialize power spectrum model objects and fit the power spectra
        fm_B = FOOOF(aperiodic_mode='fixed')
        fm_B.fit(np.array(freqs_select), psds_B_id_select)
        fm_B.plot()
        plt.title(p_id + args.conditions[0] + " fit")
        pdf.savefig()
        plt.close()

        if nr_cond == 2:
            fm_A = FOOOF(aperiodic_mode='fixed')
            fm_A.fit(np.array(freqs_select), psds_A_id_select)
            fm_A.plot()
            plt.title(p_id + args.conditions[0] + " fit")
            pdf.savefig()
            plt.close()

        # Aperiodic parameters
        offset_Base_fooof.append(fm_B.aperiodic_params_[0])
        exponent_Base_fooof.append(fm_B.aperiodic_params_[1]*-1)

        if nr_cond == 2:
            # make the foof_exponent *-1 to fit linear one
            offset_Anes_fooof.append(fm_A.aperiodic_params_[0])
            exponent_Anes_fooof.append(fm_A.aperiodic_params_[1]*-1)

        #   Calculate Aperiodic signal Linear Regression
        freqs_select_log = np.array(np.log10(freqs_select))

        lm_B = LinearRegression()
        lm_B.fit(freqs_select_log.reshape(-1,1), np.log10(psds_B_id_select))

        if nr_cond == 2:
            lm_A = LinearRegression()
            lm_A.fit(freqs_select_log.reshape(-1,1), np.log10(psds_A_id_select))
    

        # Aperiodic parameters
        offset_Base_lin.append(lm_B.intercept_)
        exponent_Base_lin.append(lm_B.coef_[0])

        if nr_cond == 2:
            offset_Anes_lin.append(lm_A.intercept_)
            exponent_Anes_lin.append(lm_A.coef_[0])

    # reduce info by the missing data
    info = info[np.invert(np.isin(P_IDS,missing_ID_B))]

    toplot = pd.DataFrame()
    toplot['ID'] = info['Patient']
    toplot['outcome'] = info['Outcome'].astype(int)
    try:
        toplot['scale'] = info['CRSR']
    except:
        toplot['scale'] = info['GCS_sedoff']
    toplot['Group'] = info['Group']
    toplot['Age'] = info['Age'].astype(int)
    toplot['exponent_Base_lin']=exponent_Base_lin
    toplot['exponent_Base_fooof']=exponent_Base_fooof
    toplot['offset_Base_lin']=offset_Base_lin
    toplot['offset_Base_fooof']=offset_Base_fooof
    if nr_cond == 2:
        toplot['exponent_Anes_lin'] = exponent_Anes_lin
        toplot['exponent_Anes_fooof'] = exponent_Anes_fooof
        toplot['offset_Anes_lin'] = offset_Anes_lin
        toplot['offset_Anes_fooof'] = offset_Anes_fooof
        toplot['diff_exponent_lin'] = toplot['exponent_Base_lin']-toplot['exponent_Anes_lin']
        toplot['diff_exponent_fooof'] = toplot['exponent_Base_fooof']-toplot['exponent_Anes_fooof']
        toplot['diff_offset_lin'] = toplot['offset_Base_lin']-toplot['offset_Anes_lin']
        toplot['diff_offset_fooof'] = toplot['offset_Base_fooof']-toplot['offset_Anes_fooof']

    fig = plot_correlation(toplot, 'exponent_Base_fooof', 'exponent_Base_lin')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    fig = plot_correlation(toplot, 'offset_Base_fooof', 'offset_Base_lin')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    #fig = plot_correlation(toplot, 'scale', 'Age')
    #pdf.savefig(fig, bbox_inches='tight')
    #plt.close()

    if nr_cond == 2:
        plt.figure()
        for i in range(len(P_IDS)):
            plt.plot([1, 2], [toplot['exponent_Base_fooof'][i], toplot['exponent_Anes_fooof'][i]], 'r')
            plt.plot([1, 2], [toplot['exponent_Base_lin'][i], toplot['exponent_Anes_lin'][i]], '--g')
        plt.xticks([1, 2], [args.conditions[0], args.conditions[1]])
        plt.title("Slope (red fooof, green linear)")
        pdf.savefig()
        plt.close()

    if len(np.unique(toplot['Group'])) > 1:
        plot_group_correlations(data=toplot, start=5, category='outcome', group=group, pdf=pdf)

    toplot.to_csv("{}/aperiodic_signal_{}_{}.csv".format(output_dir, frequency_range[0],frequency_range[1]))

    pdf.close()

