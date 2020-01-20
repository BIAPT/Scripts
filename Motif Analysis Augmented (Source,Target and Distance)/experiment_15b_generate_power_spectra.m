%{
    Yacine Mahdid 2020-01-08
    This script is used to generate power spectra for the MDFA dataset.
    This was requested by the reviewer to make sure that the motif are not
    epiphenomenon of the shift in alpha power.
%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

power_output_path = mkdir_if_not_exist(output_path,'power');
% Iterate over the participants
for p = 1:length(participants)

    % Create the participants directory
    participant = participants{p};
    power_participant_output_path =  mkdir_if_not_exist(power_output_path, participant);

    % Iterate over the states
    for s = 1:length(states)
        state = states{s};
        
        % Load the recording
        raw_data_filename = strcat(participant,'_',state,'.set');
        data_location = strcat(raw_data_path,filesep,participant);
        recording = load_set(raw_data_filename,data_location);
        
        % Calculate power
        power_state_filename = strcat(power_participant_output_path,filesep,state,'_power.mat');
        
        % setup the dynamic parameters
        window_size = floor(recording.length_recording / recording.sampling_rate); % in seconds
        step_size = window_size; 
        result_td = na_topographic_distribution(recording, window_size, step_size, power_param.bandpass);
        [power, location] = filter_non_scalp_vector(result_td.data.power, result_td.metadata.channels_location);
       
        save(power_state_filename, 'result_td');
    end
end