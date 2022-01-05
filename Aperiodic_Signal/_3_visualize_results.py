from sklearn.linear_model import LinearRegression
import matplotlib.backends.backend_pdf as pltpdf
from fooof.objs import fit_fooof_3d, combine_fooofs
from utils.visualize import plot_group_correlations
from utils.visualize import plot_two_curves
from utils.visualize import plot_cat_curves
from utils.visualize import plot_correlation
import matplotlib.pyplot as plt
from fooof import FOOOFGroup
from fooof import FOOOF
import pandas as pd
import numpy as np
import argparse
import pickle
import mne
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize the aperiodic signal portion.')
    parser.add_argument('input_dir', type=str, action='store',
                        help='folder name containing the PSD data')
    parser.add_argument('patient_information', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('--conditions', '-cond', nargs='*', action='store', default='Baseline Anesthesia',
                        help='The "task" or conditions you want to compare for example Baseline Anesthesia'
                             'can be only Base or Baseline and Anesthesia')
    parser.add_argument('--frequency_range', '-freq', nargs='*', action='store', default='1 40',
                        help='The freqency band to calculate the aperiodic signal on. For example 1 20')
    parser.add_argument('--electrode', '-el', nargs=1, action='store', default=['all'],
                        help='On which electrode should the alperiodic slope be computed '
                             'default is all)')
    parser.add_argument('--method', nargs=1, action='store', choices=('Multitaper','Welch'),
                        help='The method used for Spectral decomposition in Step 1')
    args = parser.parse_args()
    nr_cond = len(args.conditions)
    frequency_range = [int(args.frequency_range[0]), int(args.frequency_range[1])]
    method = args.method[0]
    electrode = args.electrode[0]
    input_dir = args.input_dir

    # make ouput directory
    output_dir = os.path.join(input_dir, 'aperiodic_signal')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = pltpdf.PdfPages("{}/summary_aperiodic_signal_{}_{}_{}.pdf".format
                          (output_dir, frequency_range[0], frequency_range[1], electrode))

    # load patient info
    info = pd.read_csv(args.patient_information, sep = '\t')
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
    params_B_fooof = pd.read_csv('{}\Params_{}_{}_{}_{}_{}.txt'.format
                                 (output_dir, cond_B, frequency_range[0], frequency_range[1], 'fooof', electrode),sep=' ')

    params_B_lin = pd.read_csv('{}\Params_{}_{}_{}_{}_{}.txt'.format
                                 (output_dir, cond_B, frequency_range[0], frequency_range[1], 'linreg', electrode),sep=' ')

    if nr_cond == 2:
        params_A_fooof = pd.read_csv('{}\Params_{}_{}_{}_{}_{}.txt'.format
                               (output_dir, cond_A, frequency_range[0], frequency_range[1], 'fooof', electrode),
                               sep=' ')
        params_A_lin = pd.read_csv('{}\Params_{}_{}_{}_{}_{}.txt'.format
                                   (output_dir, cond_A, frequency_range[0], frequency_range[1], 'linreg', electrode),
                                   sep=' ')


    """
        1) load PSD data
    """
    # load power spectral data
    PSD_B = pickle.load(open("{}/{}/PSD_{}_{}.pkl".format(args.input_dir, cond_B, method, cond_B), "rb"))
    PSD_B_ID = PSD_B[-1]
    P_ID_B = params_B_lin['ID']
    # load frequencies
    datapath = os.path.join(args.input_dir, cond_B, 'Frequency_{}_{}.txt'.format(method, cond_B))
    freqs_B = pd.read_csv(datapath, sep=' ', header=None)
    freqs_B = np.squeeze(freqs_B)

    psd_B = []
    for i, p_id in enumerate(P_ID_B):
        # select individual PSD values, depending on ID
        index_id = np.where(PSD_B_ID == p_id)[0][0]
        psd_B_p = PSD_B[index_id]
        psd_B_mean = np.mean(np.mean(psd_B_p, axis=0), axis=0)
        psd_B.append(psd_B_mean)

    psd_B = pd.DataFrame(psd_B)
    psd_B['ID'] = P_ID_B
    info_B = info[np.isin(P_IDS, P_ID_B)]
    info_B = info_B.reset_index()
    outcome_B = info_B['Outcome']
    group_B = info_B['Group']


    if nr_cond == 2:
        PSD_A = pickle.load(open("{}/{}/PSD_{}_{}.pkl".format(args.input_dir, cond_A, method, cond_A), "rb"))
        PSD_A_ID = PSD_A[-1]
        P_ID_A = params_A_lin['ID']
        # load frequencies
        datapath = os.path.join(args.input_dir, cond_A, 'Frequency_{}_{}.txt'.format(method, cond_A))
        freqs_A = pd.read_csv(datapath, sep=' ', header=None)
        freqs_A = np.squeeze(freqs_A)

        psd_A = []
        for i, p_id in enumerate(P_ID_A):
            # select individual PSD values, depending on ID
            index_id = np.where(PSD_B_ID == p_id)[0][0]
            psd_A_p = PSD_A[index_id]
            psd_A_mean = np.mean(np.mean(psd_A_p, axis=0), axis=0)
            psd_A.append(psd_A_mean)

        psd_A = pd.DataFrame(psd_A)
        psd_A['ID'] = P_ID_A
        info_A = info[np.isin(P_IDS, P_ID_A)]
        info_A = info_A.reset_index()
        outcome_A = info_A['Outcome']
        group_A = info_A['Group']

    if nr_cond == 2:
        # plot both signals averaged over time
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
                          title='{} PSD_el_{}'.format(args.conditions[0], electrode),
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



    # accounts for the case that some are in cond A but not B
    keep_A = np.isin(P_ID_A, P_ID_B)
    keep_B = np.isin(P_ID_B, P_ID_A)
    keep = np.isin(P_IDS, P_ID_B)

    toplot = pd.DataFrame()
    toplot['ID'] = P_ID_B[keep_B]
    toplot['outcome'] = outcome_B[keep_B].astype(int)
    toplot['scale'] = (info['CRSR'][keep]).reset_index(drop=True)
    toplot['Group'] = info['Group'][keep].reset_index(drop=True)
    toplot['Age'] = info['Age'][keep].astype(int).reset_index(drop=True)
    toplot['exponent_Base_lin'] = params_B_lin['exponent']
    toplot['exponent_Base_fooof'] = params_B_fooof['exponent']
    toplot['offset_Base_lin'] = params_B_lin['offset']
    toplot['offset_Base_fooof'] = params_B_fooof['offset']
    if nr_cond == 2:
        toplot['exponent_Anes_lin'] = params_A_lin['exponent']
        toplot['exponent_Anes_fooof'] = params_A_fooof['exponent']
        toplot['offset_Anes_lin'] = params_A_lin['offset']
        toplot['offset_Anes_fooof'] = params_A_fooof['offset']
        toplot['diff_exponent_lin'] = toplot['exponent_Base_lin'] - toplot['exponent_Anes_lin']
        toplot['diff_exponent_fooof'] = toplot['exponent_Base_fooof'] - toplot['exponent_Anes_fooof']
        toplot['diff_offset_lin'] = toplot['offset_Base_lin'] - toplot['offset_Anes_lin']
        toplot['diff_offset_fooof'] = toplot['offset_Base_fooof'] - toplot['offset_Anes_fooof']

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
        for i in toplot.index:
            plt.plot([1, 2], [toplot['exponent_Base_fooof'][i], toplot['exponent_Anes_fooof'][i]], 'r')
            plt.plot([1, 2], [toplot['exponent_Base_lin'][i], toplot['exponent_Anes_lin'][i]], '--g')
        plt.xticks([1, 2], [args.conditions[0], args.conditions[1]])
        plt.title("Slope (red fooof, green linear)")
        pdf.savefig()
        plt.close()

    if len(np.unique(toplot['Group'])) > 1:
        plot_group_correlations(data=toplot, start=5, category='outcome', group=group, pdf=pdf)

    toplot.to_csv("{}/aperiodic_signal_{}_{}.csv".format(output_dir, frequency_range[0], frequency_range[1]))

    pdf.close()
