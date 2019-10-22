%% Loading the .SET file for unconsciousness
[filepath,name,ext] = fileparts(mfilename('fullpath'));


participant_name = 'wsas09_recovery';
participant_path = strcat('/home/yacine/Documents/DOC Features Over Time/data');

recording = load_set(strcat(participant_name,'.set'),participant_path);

%% Running wPLI with small step size

% wPLI
% parameters
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = 0.1; % in seconds
wsas.result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
