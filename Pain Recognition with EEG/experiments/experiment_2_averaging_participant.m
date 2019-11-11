%{
    This script was written by Yacine Mahdid 2019-11-10 for the analysis of
    the Pain EEG data collected from the Shrinner hospital.
    Here we are using no_pain and hot1 values
%}
%% Make a script to iterate over the healthy participants folder
% Setting up path variables
base_dir = "/home/yacine/Documents/pain_and_eeg/results/healthy";
%% Setting up experiment variables (this will be shipped inside the helper function)
% The variables are in the calculate_features function

% The participants folder are named HE001 to HE014
% we can generate them like this  sprintf('%03d',participant_id)
num_participant = 12; % we have 12 because after that we have different labeling
participant_label = cell(num_participant,1);
participant_path = cell(num_participant,1);
for p_id = 1:num_participant
    participant_label{p_id} = sprintf('HE%03d',p_id);
    participant_path{p_id} = sprintf('%s/%s',base_dir,participant_label{p_id});
end

%% Iterate over all the participant and gather the baseline and pain
result = struct();

for p_id = 1:num_participant
    disp(sprintf("Analyzing participant: %s",participant_label{p_id}));
    % Create the path
    data_path = sprintf('%s/%s.mat',base_dir,participant_label{p_id});
    
    % Load the data
    data = load(data_path);
    data = data.result;
    
    % Average out the windows
    result.baseline = average_windows(data.healthy);
    result.hot_pain = average_windows(data.hot_pain);
    

end

% Save these average participant to the output directory
output_path = sprintf('%s/HEAVG.mat',base_dir);
save(output_path, 'result')

function [result] = average_windows(data)
    % Average out the spectral power
    
end
