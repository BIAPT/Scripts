% Hub location parameters
t_level_wpli = 0.10; % keep top 10% of the data
t_level_hub = 0.10;

% Get the relevant data out
wpli_windows = result_wpli.data.wpli;
location = result_wpli.metadata.channels_location;

[num_window, num_channels, ~] = size(wpli_windows);
median_locations = zeros(1,num_window);
mean_locations = zeros(1,num_window);
for w = 1:num_window
    curr_wpli = squeeze(wpli_windows(w,:,:));
    [filt_wpli, filt_location] = filter_non_scalp(curr_wpli, location);
    
    % Calculate hub location for this sub wpli network
    % Threshold at 10%
    b_wpli = binarize_matrix(threshold_matrix(filt_wpli, t_level_wpli));
    [median_locations(w), mean_locations(w), channels_degree] = binary_hub_location(b_wpli, location, t_level_hub);
end


figure;
plot(median_locations);
hold on;
plot(mean_locations);
legend('median','mean');