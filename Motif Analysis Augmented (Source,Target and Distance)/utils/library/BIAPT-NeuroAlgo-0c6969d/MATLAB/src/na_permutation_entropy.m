function [result] = na_permutation_entropy(recording, frequency_band, window_size, step_size, embedding_dimension, time_lag )
    %NA_PERMUTATION_ENTROPY NeuroAlgo implementation of wpli that works with Recording
    % NOTE: right now we are only doing non-overlapping window (in sec)
    % NOTE: We are also only doing fullband eeg
    
    %% Getting configuration
    configuration = get_configuration();
    
    %% Setting Result
    result = Result('permutation entropy', recording);
    result.parameters.frequency_band = frequency_band;
    result.parameters.window_size = window_size;
    result.parameters.embedding_dimension = embedding_dimension;
    result.parameters.time_lag = time_lag;
    
    %% Variable Initialization
    channels_location = recording.channels_location;
    
    %% Filtering the data
    print_message(strcat("Filtering Data from ",string(frequency_band(1)), "Hz to ", string(frequency_band(2)), "Hz."),configuration.is_verbose);
    [recording] = recording.filter_data(recording.data, frequency_band);
    
    % Here we init the sliding window slicing 
    recording = recording.init_sliding_window(window_size, step_size);
    number_window = recording.max_number_window;
    
    % if we don't have channels location we skip the part below
    if(~isempty(recording.channels_location))
        anterior_mask = ([channels_location.is_anterior] == 1);
        posterior_mask = ([channels_location.is_posterior] == 1);
    end

    
    %% Calculation on the windowed segments
    result.data.permutation_entropy = zeros(number_window, recording.number_channels);
    result.data.normalized_permutation_entropy = zeros(number_window, recording.number_channels);
    result.data.avg_permutation_entropy_anterior = zeros(1,number_window);
    result.data.avg_permutation_entropy_posterior = zeros(1,number_window); 
    
    for i = 1:number_window
       print_message(strcat("Permutation Entropy at window: ",string(i)," of ", string(number_window)),configuration.is_verbose); 
       [recording, segment_data] = recording.get_next_window();
       
       % Calculate the permutation entropy
       [pe, normalized_pe] = permutation_entropy(segment_data, embedding_dimension, time_lag); 
       result.data.permutation_entropy(i,:,:) = pe;
       
       % Saving the information
       result.data.normalized_permutation_entropy(i,:,:) = normalized_pe;
       % if we don't have channels location we skip the part below
       if(isempty(recording.channels_location))
           continue
       end
       
       result.data.avg_permutation_entropy_anterior(i) = mean(normalized_pe(anterior_mask));
       result.data.avg_permutation_entropy_posterior(i) = mean(normalized_pe(posterior_mask));
    end
end

