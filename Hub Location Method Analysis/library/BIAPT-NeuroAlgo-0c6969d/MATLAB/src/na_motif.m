function [result] = na_motif(recording, frequency_band, window_size, step_size, number_rand_network, bin_swaps, weight_frequency)
%NA_MOTIF Summary of this function goes here
%   Detailed explanation goes here
    
    %% Getting the configuration
    configuration = get_configuration();
    
    %% Setting the Result
    result = Result('motif', recording);
    result.parameters.frequency_band = frequency_band;
    result.parameters.window_size = window_size;
    result.parameters.step_size = step_size;    
    result.parameters.number_rand_network = number_rand_network;
    result.parameters.bin_swaps = bin_swaps;
    result.parameters.weight_frequency = weight_frequency;
    
    %% Filtering the data
    print_message(strcat("Filtering Data from ",string(frequency_band(1)), "Hz to ", string(frequency_band(2)), "Hz."),configuration.is_verbose);
    [recording] = recording.filter_data(recording.data, frequency_band);
    
    % Here we init the sliding window slicing 
    recording = recording.init_sliding_window(window_size, step_size);
    number_window = recording.max_number_window;
    number_channels = recording.number_channels;
    
    
    %% Calculation on the windowed segments
    %intensity, coherence, frequency, norm_intensity, norm_coherence, norm_frequency
    init_mat = zeros(13,number_channels,number_window);
    result.data.intesity = init_mat;
    result.data.coherence = init_mat;
    result.data.frequency = init_mat;
    
    result.data.norm_intesity = init_mat;
    result.data.norm_coherence = init_mat;
    result.data.norm_frequency = init_mat;
    
    for i = 1:number_window
       print_message(strcat("Motif(3) at window: ",string(i)," of ", string(number_window)),configuration.is_verbose); 
       [recording, segment_data] = recording.get_next_window();
       
       % Calculating hub data for the segment
       % TODO
       % Saving the hub data for this segment

    end
end

