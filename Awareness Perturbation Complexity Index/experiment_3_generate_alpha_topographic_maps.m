%{
    Goal of this experiment is to generate from the .set files 3 sets of
    topographic maps. One for the baseline, one for the anesthesia and one
    for the recovery for now.
    -> Here we want to do only alpha and create a vector per maps that
    correspond to each channels.
    -> We also need to calculate the power on a small window (1 seconds for
    now)
    -> We are using NeuroAlgo for this
%}

%% Load the data
data_path = 'C:\Users\biapt\Documents\GitHub\NeuroAlgo\MATLAB\example\test_data';
test_recording = load_set('test_data.set',data_path);

%% Calculate a test Topographic Distribution (TD)
window_size = 10; % in seconds
step_size = 0.1; % in seconds
result_td = na_topographic_distribution(test_recording, window_size, step_size);
% Note: Power is (number_channels, number_frequency) which goes from 1Hz to
% sampling_rate/2 (Nyquist frequency)

%% Average the data from a particular frequency and store it into topographic_maps
bandpass = [8 13]; % alpha
[number_maps, number_channels, number_frequencies] = size(result_td.data.power);
topographic_maps = zeros(number_maps,number_channels);

% Average the power at a specific bandpass
for m = 1:number_maps
    for c = 1:number_channels
        
        % Average the power at this particular channel in our bandpass
        channel_power = result_td.data.power(m,c,bandpass(1):bandpass(2));
        topographic_maps(m,c) = mean(channel_power);
        
    end
end

%% Make a video out of this to visualize if we are doing ok
full_path = 'C:\Users\biapt\Documents\GitHub\Scripts\Awareness Perturbation Complexity Index\test_video';
channels_location = result_td.metadata.channels_location;
make_video_topographic_map(full_path, topographic_maps, channels_location, step_size)