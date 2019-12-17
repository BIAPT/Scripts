%Danielle Nadin 11-12-2019
%Sweep through range of network thresholds and compute binary small-worldness to determine 
% the 'small-world regime range' as defined in Basset et al (2008). 

% modified by Yacine Mahdid 2019-12-12
%
% Experiment Variables

filename = 'MDFA17_BASELINE.set';
filepath = 'C:\Users\biapt\Desktop\motif fix\mdfa17_data';
recording = load_set(filename, filepath);


% wPLI Properties
frequency_band = [8 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

pli_matrix = result_wpli.data.avg_wpli;
channels_location = recording.channels_location;
threshold_range = 0.90:-0.01:0.01; % More connected to less connected

% Here we need to filter the non_scalp channels
[pli_matrix,channels_location] = filter_non_scalp(pli_matrix,channels_location);

%loop through thresholds
for j = 1:length(threshold_range) 
    current_threshold = threshold_range(j);
    disp(strcat("Doing the threshold : ", string(current_threshold)));
    
    % Thresholding and binarization using the current threshold
    t_network = threshold_matrix(pli_matrix, current_threshold);
    b_network = binarize_matrix(t_network);
    
    % check if the binary network is disconnected
    % Here our binary network (b_network) is a weight matrix but also an
    % adjacency matrix.
    distance = distance_bin(b_network);
    
    % Here we check if there is one node that is disconnected
    if(sum(isinf(distance(:))))
        disp(strcat("Final threshold: ", string(threshold_range(j-1))));
        break;
    end
end