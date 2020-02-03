function [filtered_data] = filter_bandpass(data, sampling_frequency, low_frequency, high_frequency)
%FILTER_BANDPASS Summary of this function goes here
%   Detailed explanation goes here
    data = double(data)'; % HERE CHANGED THIS THING RIGHT HERE '
            
    %% Design filter and filter the data
    [b,a]=butter(1, [low_frequency/(sampling_frequency/2) high_frequency/(sampling_frequency/2)]);
    filtered_data=filtfilt(b,a,data);
    filtered_data = filtered_data';
end

