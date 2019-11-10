%{
    This script was written by Yacine Mahdid 2019-11-10 for the analysis of
    the Pain EEG data collected from the Shrinner hospital.
    Here we are using no_pain and hot1 values
%}
%% Make a script to iterate over the healthy participants folder
% Setting up path variables
base_dir = "/home/yacine/Documents/Cleaned_Healthy_EEG";
output_dir = "/home/yacine/Documents/results/healthy";
%% Setting up experiment variables (this will be shipped inside the helper function)
% The variables are in the calculate_features function

% The participants folder are named HE001 to HE014
% we can generate them like this  sprintf('%03d',participant_id)
num_participant = 14;
participant_label = cell(num_participant,1);
participant_path = cell(num_participant,1);
for p_id = 1:num_participant
    participant_label{p_id} = sprintf('HE%03d',p_id);
    participant_path{p_id} = sprintf('%s/%s',base_dir,participant_label{p_id});
end

%% Iterate over all the participant and gather the baseline and pain
for p_id = 1:num_participant
    
    % Create the path
    baseline_name = sprintf('%s_nopain.set',participant_label{p_id});
    hot_pain_name = sprintf('%s_hot1.set',participant_label{p_id});
    
    % Load the data
    baseline_recording = load_set(baseline_name, participant_path{p_id});
    hot_pain_recording = load_set(hot_pain_name, participant_path{p_id});
    
    % Calculate some features
    result = struct();
    result.healthy = calculate_features(baseline_recording);
    result.hot_pain = calculate_features(hot_pain_recording);
    
    % Save these feature to the output directory
    output_path = sprintf('%s/%s.mat',output_dir,participant_label{p_id});
    save(output_path, 'result')
end

function [result] = calculate_features(recording)
    alpha_band = [8 13];
    % Spectrogram
    % TODO: Rewrite this one

    % Topographic Map
    tm_window_size = 10; % in seconds
    tm_step_size = tm_window_size; % in seconds
    tm_bandpass = alpha_band; % in Hz

    % Permutation Entropy
    pe_frequency_band = alpha_band; % This is in Hz
    pe_window_size = 10; % This is in seconds and will be how we chunk the whole dataset
    embedding_dimension = 5;
    time_lag = 4;

    % wPLI & dPLI
    pli_frequency_band = alpha_band; % This is in Hz
    pli_window_size = 10; % This is in seconds and will be how we chunk the whole dataset
    number_surrogate = 10; % Number of surrogate wPLI to create
    p_value = 0.05; % the p value to make our test on
    step_size = pli_window_size;
end

