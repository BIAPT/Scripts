function bins = binit(data_in, N)
% Usage bins = binit(data_in)
%
% Split the signal in N equal length samples and average the value over
% each of these segments. Returns an array containing the values of each
% averages bin.
%
% input: data_in - signal
%           N    - number of segment desired
%
% Output: bins - N 
%
% Coded by: Pascal Fortin
% Supervised by: Stefanie Blain-Moraes, Jeremy R. Cooperstock
% McGill University, Montreal, Canada
% 
% Detailed description
% 1. Segment the signal in N equal length parts
% 2. Compute average signal on each segment

    % Length of the signal
    signal_length = length(data_in);
    % Compute length of each segment
    segment_length = floor(signal_length/N);

    bins = mean(data_in(1:segment_length));
    for n = 1:N-1
        bins(n+1) = mean(data_in(n*segment_length:(n+1)*segment_length)); 
    end

end