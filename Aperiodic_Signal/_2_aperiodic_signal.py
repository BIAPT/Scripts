#!/usr/bin/env python

from sklearn.linear_model import LinearRegression
from fooof.objs import fit_fooof_3d
from fooof import FOOOFGroup
import pandas as pd
import numpy as np
import argparse
import pickle
import mne
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate the aperiodic signal portion using different Models.')
    parser.add_argument('input_dir', type=str, action='store',
                        help='folder name containing the psd pickles from step 1')
    parser.add_argument('data_dir', type=str, action='store',
                        help='folder name containing the data in .fif format')
    parser.add_argument('patient_information', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('--condition', '-cond', type=str, action='store',
                        help='The "task" or conditions you want to caluclate for example Baseline or Anesthesia')
    parser.add_argument('--frequency_range', '-freq', nargs='*', action='store', default='1 40',
                        help='The freqency band to calculate the aperiodic signal on. For example 1 20')
    parser.add_argument('--method', nargs=1, action='store', default=['Multitaper'], choices=('Multitaper','Welch'),
                        help='The method used for Spectral decomposition in Step 1')
    parser.add_argument('--electrode', '-el', nargs=1, action='store', default=['all'],
                        help='On which electrode should the alperiodic slope be computed '
                             'default is all)')
    parser.add_argument('--model', '-model', nargs=1, action='store', default=['fooof'],
                        help='Can be the fooof model or linreg',choices=('fooof','linreg'))

    # read out arguments
    args = parser.parse_args()
    electrode = args.electrode
    nr_cond = len(args.condition)
    frequency_range = [int(args.frequency_range[0]), int(args.frequency_range[1])]
    method = args.method[0]
    cond = str(args.condition)
    model = str(args.model[0])

    # make ouput directory
    output_dir = os.path.join(args.input_dir, 'aperiodic_signal')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load patient info
    info = pd.read_csv(args.patient_information,sep = '\t')
    P_IDS = info['Patient']
    outcome = info['Outcome']
    group = info['Group']

    # define empty DataFrames to save Baseline data
    offset = []
    exponent = []
    offset_space = []
    exponent_space = []
    missing_ID = []

    # load power spectral data
    PSD = pickle.load(open("{}/{}/PSD_{}_{}.pkl".format(args.input_dir, cond, method, cond), "rb"))
    PSD_ID = PSD[-1]

    # load frequencies
    datapath = os.path.join(args.input_dir, cond, 'Frequency_{}_{}.txt'.format(method, cond))
    freqs = pd.read_csv(datapath, sep=' ', header=None)
    freqs = np.squeeze(freqs)

    # prepare empty dataframe for PSD per patient and selected average
    # size electrodes * patients
    psds = []

    for p_id in P_IDS:
        not_missing = True
        # select individual PSD values, depending on ID
        index_id = np.where(PSD_ID == p_id)[0][0]
        PSD_p = PSD[index_id]

        """
        1) LOAD RAW DATA 
        """
        # load the epoched raw data (needed later for selection of the electrode)
        input_fname = "{}/sub-{}/eeg/epochs_{}_{}.fif".format(args.data_dir, p_id, p_id, cond)
        raw = mne.read_epochs(input_fname)
        # remove channels marked as bad and non-brain channels
        raw.drop_channels(raw.info['bads'])

        # load frequencies
        datapath = os.path.join(args.input_dir, cond, 'Frequency_{}_{}.txt'.format(method, cond))
        freqs_B = pd.read_csv(datapath, sep=' ', header=None)
        freqs_B = np.squeeze(freqs_B)

        if electrode[0] == 'all':
            # take the raw PSD data of this participant
            nr_chan = PSD_p.shape[1]
            nr_time = PSD_p.shape[0]

        # If an electrode is specified
        else:
            # search for this electrode in the given Data
            if np.isin(np.array(raw.info.ch_names), electrode).any():
                select_ch = np.where(np.array(raw.info.ch_names) == electrode)[0][0]
                # select right channel
                PSD_p = PSD_p[:,select_ch,:]
                nr_chan = 1
                nr_time = PSD_p.shape[0]

            else:
                # add this ID to the missing IDS for this electrode
                missing_ID.append(p_id)
                not_missing = False

        # only continue if electrode is in the data
        if not_missing:

            """
            #   Calculate Aperiodic signal
            """
            if model == 'linreg':
                # create empty dataframe electrode x time
                exp_p = np.empty([nr_chan, nr_time])
                offs_p = np.empty([nr_chan, nr_time])

                # loop over time and electrodes
                for c in range(nr_chan):
                    for t in range(nr_time):
                        if  electrode[0] == 'all':
                            # select only current time window and channel
                            PSD_tmp = PSD_p[t, c]
                        else:
                            # select only current time window and channel
                            PSD_tmp = PSD_p[t, :]

                        # only use defined frequency band
                        index_toselect = np.where((freqs >= frequency_range[0]) & (freqs <= frequency_range[1]))[0]
                        freqs_select = freqs[index_toselect]
                        PSD_tmp_select = PSD_tmp[index_toselect]

                        # log the frequency and PSD
                        freqs_select_log = np.array(np.log10(freqs_select))
                        PSD_log = np.log10(PSD_tmp_select)

                        #   Calculate Aperiodic signal Linear Regression Model
                        mdl = LinearRegression()
                        mdl.fit(freqs_select_log.reshape(-1, 1), PSD_log)

                        # extract Aperiodic parameters
                        offs_p[c, t] = mdl.intercept_
                        exp_p[c, t] = mdl.coef_[0]

                offset.append(np.mean(offs_p))
                exponent.append(np.mean(exp_p))

                offset_space.append(np.mean(offs_p,axis=1))
                exponent_space.append(np.mean(exp_p,axis=1))



            if model == 'fooof':
                # power spectra across data epochs within subjects, as [n_epochs, n_channels, n_freqs]
                PSD_p_3d = np.array(PSD_p)

                # make it 3-dimensional if we only have 1 electrode
                if len(PSD_p_3d.shape) == 2:
                    PSD_p_3d = PSD_p_3d.reshape(1, PSD_p_3d.shape[0], PSD_p_3d.shape[1])

                # initiate FOOOF model:
                fg = FOOOFGroup(aperiodic_mode = 'fixed')
                # Fit the 3D array of power spectra
                fgs = fit_fooof_3d(fg, np.array(freqs), PSD_p_3d, freq_range=frequency_range, n_jobs=-1)

                # Aperiodic parameters averaged over all
                offset.append(np.mean(fg.get_params('aperiodic_params', 'offset')))
                exponent.append(np.mean(fg.get_params('aperiodic_params', 'exponent')* -1))

                # Aperiodic parameters space resolved
                offset_space.append(fg.get_params('aperiodic_params', 'offset'))
                exponent_space.append(fg.get_params('aperiodic_params', 'exponent') * -1)

        print("Finished Subject  {}".format(p_id))

    params_df = pd.DataFrame()
    params_df['ID'] = P_IDS[np.invert(np.isin(P_IDS, missing_ID))]
    params_df['offset'] = offset
    params_df['exponent'] = exponent

    params_df.to_csv('{}\Params_{}_{}_{}_{}_{}.txt'.format(output_dir, cond,
                                                     frequency_range[0], frequency_range[1],
                                                     model, electrode[0]), index=False, sep=' ')


    if electrode[0] == 'all':
        params_space_df = pd.DataFrame()
        params_space_df['ID'] = P_IDS[np.invert(np.isin(P_IDS, missing_ID))]
        params_space_df['offset'] = offset_space
        params_space_df['exponent'] = exponent_space

        params_space_df.to_csv('{}\Params_space_{}_{}_{}_{}_{}.txt'.format(output_dir, cond,
                                                                     frequency_range[0], frequency_range[1],
                                                                     model, electrode[0]), index=False, sep=' ')
