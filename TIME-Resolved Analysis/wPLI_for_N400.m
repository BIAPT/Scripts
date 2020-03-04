%% Loading a .set file
% This will allow to load a .set file into a format that is amenable to analysis
% The first argument is the name of the .set you want to load and the
% second argument is the path of the folder containing that .set file
% Here I'm getting it programmatically because my path and your path will
% be different.
[filepath,name,ext] = fileparts(mfilename('fullpath'));
%data_path = strcat(filepath,'/data');
recording_anes = load_set('MDFA05_emergence_first_5min_brainonly.set',filepath);
recording_base = load_set('MDFA05_eyes_closed_1_brainonly.set',filepath);
recording_base6 = load_set('MDFA05_eyes_closed_6_brainonly.set',filepath);
recording_anes5 = load_set('MDFA05_emergence_last_5_min_Brainonly.set',filepath);

French_N400_18 = load_set('N418_clean.set',filepath);


%{ 
    The recording class is structured as follow:
    recording.data = an (channels, timepoints) matrix corresponding to the EEG
    recording.length_recoding = length in timepoints of recording
    recording.sampling_rate = sampling frequency of the recording
    recording.number_channels = number of channels in the recording
    recording.channels_location = structure containing all the data of the channels (i.e. labels and location in 3d space)
    recording.creation_data = timestamp in UNIX format of when this class was created
%}

% wPLI
frequency_band = [8 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = 0.1;

result_wpli_anes = na_wpli(recording_anes, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_base = na_wpli(recording_base, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_base6 = na_wpli(recording_base6, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_anes5 = na_wpli(recording_anes5, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_N400 = na_wpli(French_N400_18, frequency_band, window_size, step_size, number_surrogate, p_value);

result_wpli_N418_step=result_wpli_N400.data.wpli;
result_wpli_N418_avg=result_wpli_N400.data.avg_wpli;


result_wpli_anes_step=result_wpli_anes.data.wpli;
result_wpli_anes_avg=result_wpli_anes.data.avg_wpli;

result_wpli_rest_step=result_wpli_base.data.wpli;
result_wpli_rest_avg=result_wpli_base.data.avg_wpli;

result_wpli_rest6_step=result_wpli_base6.data.wpli;
result_wpli_rest6_avg=result_wpli_base6.data.avg_wpli;

result_wpli_anes5_step=result_wpli_anes5.data.wpli;
result_wpli_anes5_avg=result_wpli_anes5.data.avg_wpli;

