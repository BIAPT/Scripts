%{
    Yacine Mahdid 2020-01-20
    This is the first try of calculating the Hub location using degree on
    wPLI in healthy individual. This require NeuroAlgo v0.0.1. The first
    step is to calculate wpli with a 0.1 steps to see if we get something
    that make sense.
%}


% Setup the experiment
setup_experiments;

% Extract Needed Variables
ppt = settings.participant;
state = settings.state;
in_path = settings.raw_data_path;
out_path = settings.output_path;

% Constructing the in and out filename
in_filename = strcat(ppt, '_', state, '.mat');
out_filename = strcat(out_path, ppt,'_',state,'_wpli.mat');

recording = load_set(in_filename, in_path);

% Calculate the wPLI which is the basis for the hub location network
% This might need to be separated into its own file
frequency_band = [8 14]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = 1;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

save(output_file_path, 'result_wpli');
