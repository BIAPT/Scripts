%{
    Yacine Mahdid 2020-01-20
    This is the first try of calculating the Hub location using degree on
    wPLI in healthy individual. This require NeuroAlgo v0.0.1.
%}


raw_data_path = 'C:/Users/biapt/Desktop/motif fix/mdfa17_data';
raw_data_filename =  'MDFA17_BASELINE.set';
recording = load_set(raw_data_filename, raw_data_path);

% Calculate the wPLI which is the basis for the hub location network
frequency_band = [8 14]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

