function [result] = na_hub_location(recording, frequency_band, window_size, step_size, number_surrogate, p_value ,threshold)
%NA_HUB_LOCATION Summary of this function goes here
%   Detailed explanation goes here

    %% Getting the configuration
    configuration = get_configuration();
    
    %% Setting the Result
    result = Result('hub location', recording);
    result.parameters.frequency_band = frequency_band;
    result.parameters.window_size = window_size;
    result.parameters.number_surrogate = number_surrogate;
    result.parameters.p_value = p_value;
    result.parameters.threshold = threshold;
    result.parameters.step_size = step_size;
    
    %% Filtering the data
    print_message(strcat("Filtering Data from ",string(frequency_band(1)), "Hz to ", string(frequency_band(2)), "Hz."),configuration.is_verbose);
    [recording] = recording.filter_data(recording.data, frequency_band);
    
    % Here we init the sliding window slicing 
    recording = recording.init_sliding_window(window_size, step_size);
    number_window = recording.max_number_window;
    
    
    %% Calculation on the windowed segments
    result.data.hub_index = zeros(1,number_window);
    result.data.hub_degree = zeros(1,number_window);
    result.data.hub_relative_position = zeros(1,number_window);
    result.data.graph = zeros(number_window, recording.number_channels, recording.number_channels);
    for i = 1:number_window
       print_message(strcat("Hub Location at window: ",string(i)," of ", string(number_window)),configuration.is_verbose); 
       [recording, segment_data] = recording.get_next_window();
       
       % Calculating hub data for the segment
       [channel_index, channel_degree, relative_position, graph] = hub_location(segment_data, recording.channels_location, number_surrogate, p_value, threshold); 
       
       % Saving the hub data for this segment
       result.data.hub_index(i) = channel_index;
       result.data.hub_degree(i) = channel_degree;
       result.data.hub_relative_position(i) = relative_position;
       result.data.graph(i,:,:) = graph;
    end
end

