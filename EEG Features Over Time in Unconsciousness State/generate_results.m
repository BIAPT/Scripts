%% Loading the three data .set file
[filepath,name,ext] = fileparts(mfilename('fullpath'));


participant_name = 'wsas07_baseline';
participant_path = strcat('/home/yacine/Documents/DOC Features Over Time/data');

recording = load_set(strcat(participant_name,'.set'),participant_path);

% Saving structure
wsas = struct();

%% Running the analysis

% wPLI
% parameters
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on

wsas.result_wpli = na_wpli(recording, frequency_band, window_size, number_surrogate, p_value);

% dPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on

wsas.result_dpli = na_dpli(recording, frequency_band, window_size, number_surrogate, p_value);

% Hub Location (HL)
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
threshold = 0.10; % This is the threshold at which we binarize the graph

wsas.result_hl = na_hub_location(recording, frequency_band, window_size, number_surrogate, p_value, threshold);

% Permutation Entropy (PE)
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
embedding_dimension = 5;
time_lag = 4;
wsas.result_pe = na_permutation_entropy(recording, frequency_band, window_size ,embedding_dimension, time_lag);

% Phase Amplitude Coupling (PAC)
window_size = 10;
low_frequency_bandwith =[0.1 1];
high_frequency_bandwith = [8 13];
number_bins = 18;
wsas.result_pac = na_phase_amplitude_coupling(recording, window_size, low_frequency_bandwith, high_frequency_bandwith, number_bins);
 
% Spectral Power Ratio (SPR)
window_size = 10;
time_bandwith_product = 2;
number_tapers = 3;
spectrum_window_size = 3; % in seconds
step_size = 0.1; % in seconds

wsas.result_spr = na_spectral_power_ratio(recording, window_size, time_bandwith_product, number_tapers, spectrum_window_size, step_size);

% Topographic Distribution (TD)
window_size = 10; % in seconds
frequency = 10; % in Hz
wsas.result_td = na_topographic_distribution(recording, window_size, frequency);

%% Saving the Structure
save(strcat(participant_name,".mat"), '-struct', 'wsas')