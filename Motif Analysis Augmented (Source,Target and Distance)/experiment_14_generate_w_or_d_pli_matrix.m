%{
    Yacine Mahdid 2020-01-08
    This script will calculate the wpli and the dpli matrices (at alpha)
    that are needed to run the subsequent analysis. The parameters for the
    analysis can be found in this script

    * Warning: This experiment use the setup_experiments.m script to 
    load variables. Therefore if you are trying to edit this code and you
    don't know what a variable mean take a look at the setup_experiments.m
    script.
%}

%% Seting up the variables
clear;
setup_experiments % see this file to edit the experiments

% Create the (w/d)pli directory
wpli_output_path = mkdir_if_not_exist(output_path,'wpli');
dpli_output_path = mkdir_if_not_exist(output_path,'dpli');
% Iterate over the participants
for p = 1:length(participants)
    
    % Create the participants directory
    participant = participants{p};
    disp(strcat("Participant : ",participant));
    wpli_participant_output_path =  mkdir_if_not_exist(wpli_output_path, participant);
    dpli_participant_output_path =  mkdir_if_not_exist(dpli_output_path, participant);

    % Iterate over the states
    for s = 1:length(states)
        state = states{s};
        
        % Load the recording
        raw_data_filename = strcat(participant,'_',state,'.set');
        data_location = strcat(raw_data_path,filesep,participant);
        recording = load_set(raw_data_filename,data_location);
        
        % Calculate wpli
        wpli_state_filename = strcat(wpli_participant_output_path,filesep,state,'_wpli.mat');
        result_wpli = na_wpli(recording, wpli_param.frequency_band, ...
                              wpli_param.window_size, wpli_param.step_size, ...
                              wpli_param.number_surrogate, wpli_param.p_value);
        save(wpli_state_filename, 'result_wpli');
        
        % Calculate dpli
        dpli_state_filename = strcat(dpli_participant_output_path,filesep,state,'_dpli.mat');
        result_dpli = na_dpli(recording, dpli_param.frequency_band, ...
                              dpli_param.window_size, dpli_param.step_size, ...
                              dpli_param.number_surrogate, dpli_param.p_value);
        save(dpli_state_filename, 'result_dpli');
    end
end