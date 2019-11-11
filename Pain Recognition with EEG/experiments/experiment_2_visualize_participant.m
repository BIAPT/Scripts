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

baseline_avg_spectrum = zeros(1,167);
pain_avg_spectrum = zeros(1,167);
for p_id = 1:num_participant
    disp(sprintf("Analyzing participant: %s",participant_label{p_id}));
    if(p_id == 7)
       continue 
    end
    % Create the path
    data_path = sprintf('%s/%s.mat',base_dir,participant_label{p_id});
    
    % Load the data
    data = load(data_path);
    data = data.result;
    
    % Add up the spectrograms (and average them across time)
    baseline_avg_spectrum = baseline_avg_spectrum + mean(data.healthy.sp.data.spectrums,1);
    pain_avg_spectrum = pain_avg_spectrum + mean(data.hot_pain.sp.data.spectrums,1);
    frequencies_spectrum = data.healthy.sp.data.frequencies; % this should be the same at each iteration
end

num_participant = num_participant-1;
% Average the spectrum accumulated
result.baseline_avg_spectrum = baseline_avg_spectrum/num_participant;
result.pain_avg_spectrum = pain_avg_spectrum/num_participant;
result.frequencies_spectrum = frequencies_spectrum;

% Save these average participant to the output directory
output_path = sprintf('%s/HEAVG.mat',base_dir);
save(output_path, 'result')

