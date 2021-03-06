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
in_path = strcat(in_path, ppt, filesep);
in_filename = strcat(ppt, '_', state, '.set');
out_filename = strcat(out_path, ppt,'_',state,'_wpli.mat');

recording = load_set(in_filename, in_path);

% Calculate the wPLI which is the basis for the hub location network
frequency_band = settings.wpli.frequency_band;
window_size = settings.wpli.window_size;
step_size = settings.wpli.step_size;
number_surrogate = settings.wpli.number_surrogate;
p_value = settings.wpli.p_value;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

save(out_filename, 'result_wpli');

