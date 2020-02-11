function [windowed_data] = create_sliding_window(data, window_size, step, sampling_rate)
    % This function is to get overlapping windowed data in points
    % we assume data is a c*p matrix where c are number of channels and p
    % is the number of points
    [num_channels, num_points] = size(data);
    window_size = window_size*sampling_rate; % in points
    step = floor(step*sampling_rate);
    iterator = 1:step:(num_points - window_size);
    windowed_data = zeros(length(iterator),num_channels,window_size);
    index = 1;
    for i = 1:step:(num_points - window_size)
        windowed_data(index,:,:) = data(:,i:i+window_size-1);
        index = index + 1;
    end
end