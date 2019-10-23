function [features, labels] = pre_process_participant(filename,path)
%PRE_PROCESS_PARTICIPANT preprocessing function as built by experiment_1
%   using the filename we will pre-process the data and output a feature
%   vector
   
    %% Analysis static variables
    % window segmentation
    window_size = 10; % in second
    step = 10; % in second
    % wpli calculation
    number_surrogates = 20;
    p_value = 0.05;
    
    %% Loading the participant
    fullpath = strcat(path,filesep,filename);
    participant = load(fullpath);

    %% Getting the data structure out of the .mat file
    data = participant.data;
    labels = participant.labels;
    
    %% Setup for dataset specific variables
    sampling_frequency = 128; % in Hz
    baseline_length = 3; % in seconds
    num_left_over_connection = 496; % this is removing redundant information + diagonal
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
    % wPLI calculation
    participant_wpli = zeros(num_trial, num_window, num_left_over_connection);
    for t_index = 1:num_trial
       disp(strcat("Analyzing trial #",string(t_index)));
       for w_index = 1:num_window
            disp(strcat("at window #",string(w_index)));
            % Calculating wpli on each window of every trial
            current_data = squeeze(augmented_data(t_index,w_index,:,:));
            participant_wpli(t_index,w_index,:) = reduce_matrix(wpli(current_data, number_surrogates, p_value));
       end
    end

    % At this point we have a  trial*num_window*(channel*channel) matrice which
    % will be training for one participant.
    % We will try to do a regression on the actual value of valence and arousal
    % given by each participant.
    
    % For now we will look at the average connectivity
    features = participant_wpli;
    labels = labels;
end

% Helper function to filter some redundant information in the 32*32 matrix
function [non_redundant_result] = reduce_matrix(data)
    [num_channels, ~] = size(data);
    non_redundant_result = [];
    for channel_i = 1:num_channels
       for channel_j = (channel_i+1):num_channels
          non_redundant_result = [non_redundant_result, data(channel_i,channel_j)]; 
       end
    end
end
