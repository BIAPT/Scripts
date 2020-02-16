%% Loading a .set file
% This will allow to load a .set file into a format that is amenable to analysis
% The first argument is the name of the .set you want to load and the
% second argument is the path of the folder containing that .set file
% Here I'm getting it programmatically because my path and your path will
% be different.
[filepath,name,ext] = fileparts(mfilename('fullpath'));
test_data_path = strcat(filepath,'/test_data');
recording = load_set('test_data.set',test_data_path);

% wPLI
frequency_band = [7 13]; % This is in Hz
window_size = 20; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = 0.1;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

figure;
subplot(2,1,1)
imagesc(squeeze(mean(result_wpli.data.wpli)));
colorbar
colormap jet; 
title(strcat("Average Participant at ", string(number_surrogate), " surrogates"));
subplot(2,1,2);
imagesc(squeeze(result_wpli.data.wpli(5,:,:)));
colormap jet; 
colorbar;
title(strcat("Single participant #",string(15)," at ", string(number_surrogate), " surrogates"));

% Calculating Hub Location on the average wpli
channels_location = result_wpli.metadata.channels_location;

% calculate hub location
t_level_wpli = 0.2;
t_level_hub = 0.10;
wpli = result_wpli.data.wpli;
[num_window, num_channels,~] = size(wpli);

degree_map = zeros(num_window, num_channels);
median_location = zeros(1,num_window);
max_location = zeros(1,num_window);
for i = 1:num_window   
    % binarized the wpli matrix
    b_wpli = binarize_matrix(threshold_matrix(squeeze(wpli(i,:,:)), t_level_wpli));
    [normalized_location,previous_location, channels_degree] = binary_hub_location(b_wpli, channels_location, t_level_hub);
    degree_map(i,:) = channels_degree;
    median_location(i) = normalized_location;
    max_location(i) = previous_location;
end

filename = 'hub_location_technique_comparison';
make_video_hub_location(filename, degree_map, median_location, max_location, channels_location, step_size)


function plot_hub(normalized_location, previous_location, channels_degree, channels_location)
    % Plotting the hub location
    figure;
    subplot(1,3,1)
    topoplot(channels_degree,channels_location,'maplimits','absmax', 'electrodes', 'off');
    colormap('cool');
    colorbar;
    title('Degrees')
    subplot(1,3,2);
    bar(normalized_location);
    ylim([0 1])
    title('Current Normalized Location (median)');
    subplot(1,3,3);
    bar(previous_location);
    ylim([0 1])
    title('Previous Normalized Location (max)');
end
