%{
    Yacine Mahdid 2020-01-08
    This script will calculate the wpli and the dpli matrices (at alpha)
    that are needed to run the subsequent analysis. The parameters for the
    analysis can be found in this script

%}

%% Seting up the variables
settings = load_settings();
input_path = settings.input_path;
output_path = settings.output_path;

participants = {'MDFA17'};
states = {'BASELINE', 'IF5', 'EMF5', 'EML30','EML10','EML5', 'RECOVERY'};

%% Experimental Variables
frequency_band = [8 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;

% Create the (w/d)pli directory
wpli_output_path = mkdir_if_not_exist(output_path,'wpli');
dpli_output_path = mkdir_if_not_exist(output_path,'dpli');
% Iterate over the participants
for p = 1:length(participants)

    % Create the participants directory
    participant = participants{p};
    wpli_participant_output_path =  mkdir_if_not_exist(wpli_output_path, participant);
    dpli_participant_output_path =  mkdir_if_not_exist(dpli_output_path, participant);

    % Iterate over the states
    for s = 1:length(states)
        state = states{s};
        
        % Load the recording
        raw_data_filename = strcat(participant,'_',state,'.set');
        data_location = strcat(input_path,filesep,participant);
        recording = load_set(raw_data_filename,data_location);
        
        % Calculate wpli
        wpli_state_filename = strcat(wpli_participant_output_path,filesep,state,'_wpli.mat');
        result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
        save(wpli_state_filename, 'result_wpli');
        
        % Calculate dpli
        dpli_state_filename = strcat(dpli_participant_output_path,filesep,state,'_dpli.mat');
        result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
        save(dpli_state_filename, 'result_dpli');
    end
end