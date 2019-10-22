%{
    We have two structures per participant: data and labels.
    data is trial*channel*points
    labels is trial*(valence, arousal, dominance, liking)
    -> we don't need dominance and liking
    -> we don't need the accompagning .csv files!

    The first 32 channels are EEG, the last
%}

clear;
clc; 

%% Example participant analysis
example_participant = load('example_data.mat');

%% Getting the data structure out of the .mat file
data = example_participant.data;
labels = example_participant.labels;

%% Variable Initialization
sampling_frequency = 128; % in Hz
baseline_length = 3; % in seconds
baseline_length = baseline_length*sampling_frequency + 1; % in points
[num_trial, num_channels, num_points] = size(data);
num_eeg_channels = 32;

%% Remove the first 3 seconds of data
data = data(:,:,baseline_length:num_points);

%% Filter the data (here I am not sure at what bandpass to filter it)
% Let's do fullband for a starter (4-45Hz which is the state of the data as is)

%% Remove the non-EEG channels (channels 33 to 40)
data = data(:,1:32,:);

%% Create windowed data points
window_size = 10; % in second
step = 10; % in second
% dummy function call just to get the number of window with the current
% setup
[num_window,~,~] = size(create_sliding_window(squeeze(data(1,:,:)), window_size, step, sampling_frequency));

% Populating the 'augmented' dataset
augmented_data = zeros(num_trial,num_window, num_eeg_channels, window_size*sampling_frequency);
for t_index = 1:num_trial
    disp(strcat('Analyzing trial # ', string(t_index)));
    current_data = squeeze(data(t_index,:,:));
    augmented_data(t_index,:,:,:) = create_sliding_window(current_data, window_size, step, sampling_frequency);
end


%% Calculating functional Connectivity
% Variable setup
number_surrogates = 10;
p_value = 0.05;

% wPLI calculation
participant_wpli = zeros(num_trial, num_window, num_eeg_channels, num_eeg_channels);
for t_index = 1:num_trial
   disp(strcat("Analyzing trial #",string(t_index)));
   for w_index = 1:num_window
        disp(strcat("at window #",string(w_index)));
        % Calculating wpli on each window of every trial
        current_data = squeeze(augmented_data(t_index,w_index,:,:));
        participant_wpli(t_index,w_index,:,:) = wpli(current_data, number_surrogates, p_value);
   end
end

% At this point we have a  trial*num_window*(channel*channel) matrice which
% will be training for one participant.
% We will try to do a regression on the actual value of valence and arousal
% given by each participant.
features = reshape(participant_wpli,[num_trial, num_window, (num_eeg_channels*num_eeg_channels)]);