%% Loading a .set file
% This will allow to load a .set file into a format that is amenable to analysis
% The first argument is the name of the .set you want to load and the
% second argument is the path of the folder containing that .set file
% Here I'm getting it programmatically because my path and your path will
% be different.
[filepath,name,ext] = fileparts(mfilename('fullpath'));
test_data_path = strcat(filepath,'/test_data');
recording = load_set('test_data.set',test_data_path);
%{ 
    The recording class is structured as follow:
    recording.data = an (channels, timepoints) matrix corresponding to the EEG
    recording.length_recoding = length in timepoints of recording
    recording.sampling_rate = sampling frequency of the recording
    recording.number_channels = number of channels in the recording
    recording.channels_location = structure containing all the data of the channels (i.e. labels and location in 3d space)
    recording.creation_data = timestamp in UNIX format of when this class was created
%}

%% Running the analysis
%{
    Currently we have the following 7 features that are usable with the
    recording class: wpli, dpli, hub location, permutation entropy, phase
    amplitude coupling, spectral power ratio, topographic distribution.

    If you want to get access to the features that are used without the
    recording class take a look at the /source folder
%}

% Spectral Power
window_size = 10;
time_bandwith_product = 2;
number_tapers = 3;
spectrum_window_size = 3; % in seconds
step_size = 0.1; % in seconds
bandpass = [8 13];
result_sp = na_spectral_power(recording, window_size, time_bandwith_product, number_tapers, spectrum_window_size, bandpass,step_size);


% wPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

% dPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 1; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

% Hub Location (HL)
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
threshold = 0.10; % This is the threshold at which we binarize the graph
step_size = 10;
%result_hl = na_hub_location(recording, frequency_band, window_size, step_size, number_surrogate, p_value, threshold);

% Permutation Entropy (PE)
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
embedding_dimension = 5;
time_lag = 4;
step_size = 10;
result_pe = na_permutation_entropy(recording, frequency_band, window_size , step_size, embedding_dimension, time_lag);

% Phase Amplitude Coupling (PAC)
window_size = 10;
low_frequency_bandwith =[0.1 1];
high_frequency_bandwith = [8 13];
number_bins = 18;
step_size = 10;
result_pac = na_phase_amplitude_coupling(recording, window_size, step_size, low_frequency_bandwith, high_frequency_bandwith, number_bins);

% Spectral Power Ratio (SPR)
window_size = 10;
time_bandwith_product = 2;
number_tapers = 3;
spectrum_window_size = 3; % in seconds
step_size = 5; % in seconds
result_spr = na_spectral_power_ratio(recording, window_size, time_bandwith_product, number_tapers, spectrum_window_size, step_size);

% Topographic Distribution (TD)
window_size = 10; % in seconds
step_size = window_size; % in seconds
bandpass = [8 13]; % in Hz
result_td = na_topographic_distribution(recording, window_size, step_size, bandpass);

% Note: You can put the value you want directly into the function call,
% I've laid them out beforehand so that you understand what each value
% means