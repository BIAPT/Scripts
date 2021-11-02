% Testing scripts to make sure the function parameters didn't change
% Parameters Initialization for the testing
[filepath,name,ext] = fileparts(mfilename('fullpath'));
test_data_path = strcat(filepath,'/test_data');
recording = load_set('test_data.set',test_data_path);

%% Spectral Power Parameters Test
window_size = 10;
time_bandwith_product = 2;
number_tapers = 3;
spectrum_window_size = 3; % in seconds
step_size = window_size; % in seconds
bandpass = [0.5 50];
result_sp = na_spectral_power(recording, window_size, time_bandwith_product, number_tapers, spectrum_window_size, bandpass,step_size);


%% wPLI Parameters Test
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

%% dPLI Parameters Test
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 1; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

%% Hub Location Parameters Tests
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
threshold = 0.10; % This is the threshold at which we binarize the graph
step_size = window_size;
result_hl = na_hub_location(recording, frequency_band, window_size, step_size, number_surrogate, p_value, threshold);

%% Permutation Entropy Parameters Tests
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
embedding_dimension = 5;
time_lag = 4;
step_size = window_size;
result_pe = na_permutation_entropy(recording, frequency_band, window_size , step_size, embedding_dimension, time_lag);

%% Phase Amplitude Coupling Parameters Tests
window_size = 10;
low_frequency_bandwith = [0.1 1];
high_frequency_bandwith = [8 13];
number_bins = 18;
step_size = window_size;
result_pac = na_phase_amplitude_coupling(recording, window_size, step_size, low_frequency_bandwith, high_frequency_bandwith, number_bins);

%% Spectral Power Ratio Parameters Tests
window_size = 10;
time_bandwith_product = 2;
number_tapers = 3;
spectrum_window_size = 3; % in seconds
step_size = window_size; % in seconds
result_spr = na_spectral_power_ratio(recording, window_size, time_bandwith_product, number_tapers, spectrum_window_size, step_size);

%% Topographic Distribution Parameters Tests
window_size = 10; % in seconds
step_size = window_size; % in seconds
bandpass = [8 13]; % in Hz
result_td = na_topographic_distribution(recording, window_size, step_size, bandpass);