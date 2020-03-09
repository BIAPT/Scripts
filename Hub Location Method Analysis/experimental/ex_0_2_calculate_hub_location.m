%{
    Yacine Mahdid 24/02/2020
    This script was done in order to check out if different definition of
    hubs leads to more stable hub positioning (mostly for the Clowning
    Around with the Sound of Consciousness project)

    NOTE: Separated binary_hub_location from neuroalgo into two separate
    functions
%}

setup_experiments;

load_path = strcat(settings.output_path,settings.participant,'_');


% Hub location parameters
t_level_wpli = 0.35; % keep top 10% of the data
t_level_hub = 0.10; % definition of a hub

% Get the relevant data out

data = load(strcat(load_path,'BASELINE_wpli.mat'));
baseline_wpli = data.result_wpli.data.avg_wpli;
baseline_location = data.result_wpli.metadata.channels_location;

data = load(strcat(load_path,'EMF5_wpli.mat'));
anesthesia_wpli = data.result_wpli.data.avg_wpli;
anesthesia_location = data.result_wpli.metadata.channels_location;

data = load(strcat(load_path,'RECOVERY_wpli.mat'));
recovery_wpli = data.result_wpli.data.avg_wpli;
recovery_location = data.result_wpli.metadata.channels_location;

median_locations = zeros(1, 3);
max_locations = zeros(1, 3);
bc_locations = zeros(1,3);

%% Filtering out the non-scalp channels
[baseline_wpli, baseline_location] = filter_non_scalp(baseline_wpli, baseline_location);
[anesthesia_wpli, anesthesia_location] = filter_non_scalp(anesthesia_wpli, anesthesia_location);
[recovery_wpli, recovery_location] = filter_non_scalp(recovery_wpli, recovery_location);

%% Calculate the Degree Defined Hub Location
% We are using either the median location or the max location
% Calculate hub location for this sub wpli network
% Threshold at 10%
% Baseline
baseline_wpli = binarize_matrix(threshold_matrix(baseline_wpli, t_level_wpli));
max_locations(1) = max_degree_hub_location(baseline_wpli, baseline_location);
median_locations(1) = median_degree_hub_location(baseline_wpli, baseline_location, t_level_hub);
bc_locations(1) = betweeness_hub_location(baseline_wpli, baseline_location);

% Anesthesia
anesthesia_wpli = binarize_matrix(threshold_matrix(anesthesia_wpli, t_level_wpli));
max_locations(2) = max_degree_hub_location(anesthesia_wpli, anesthesia_location);
median_locations(2) = median_degree_hub_location(anesthesia_wpli, anesthesia_location, t_level_hub);
bc_locations(2) = betweeness_hub_location(anesthesia_wpli, anesthesia_location);


% Recovery
recovery_wpli = binarize_matrix(threshold_matrix(recovery_wpli, t_level_wpli));
max_locations(3) = max_degree_hub_location(recovery_wpli, recovery_location);
median_locations(3) = median_degree_hub_location(recovery_wpli, recovery_location, t_level_hub);
bc_locations(3) = betweeness_hub_location(recovery_wpli, recovery_location);

figure;
plot(median_locations);
hold on;
plot(max_locations);
hold on;
plot(bc_locations)
legend('median','max', 'bc');


function [hub_location] = betweeness_hub_location(b_wpli, location)
%BETWEENESS_HUB_LOCATION select a channel which is the highest hub based on
%betweeness centrality and degree
% b_wpli: binary matrix
% location: 3d channels location


    
    %% 1.Calculate the degree for each electrode.
    degrees = degrees_und(b_wpli);
    norm_degree = (degrees - mean(degrees)) / std(degrees);
    a_degree = 1.0;
    
    
    %% 2. Calculate the betweeness centrality for each electrode.
    bc = betweenness_bin(b_wpli);
    norm_bc = (bc - mean(bc)) / std(bc);
    a_bc = 1.0;
    
    
    %% 3. Combine the two metric (here we assume equal weight on both the degree and the betweeness centrality)
    weights = a_degree*norm_degree + a_bc*norm_bc;
    
    %% 4.Look at the degree + betweeness centrality of the neighbouring electrodes, and if they weren't high as well, exclude this electrode as a potential hub.
    % WON'T FIX FOR NOW
    
    %% Obtain a metric for the channel that is most likely the hub epicenter
    [~, channel_index] = max(weights);
    hub_location = threshold_anterior_posterior(channel_index, location);

end

function [max_degree] = max_degree_hub_location(b_matrix, location)
%MAX_DEGREE_HUB_LOCATION choose from the channels the channel with the highest
%degree
% Input:
%   b_matrix: a binary undirected matrix
%   location: the eeg channels location in 3d space
    
    %% Caculate the unweighted degree of the network
    channels_degree = degrees_und(b_matrix);
    
    %% Calculate previous location
    [~, channel_index] = max(channels_degree);
    max_degree = threshold_anterior_posterior(channel_index, location);
end

function [median_degree] = median_degree_hub_location(b_matrix, location, t_level)
%MEDIAN_DEGREE_HUB_LOCATION choose from the the channels the channel with the highest
%degree
% Input:
%   b_matrix: a binary undirected matrix
%   location: the eeg channels location in 3d space
%   t_level: the top x percentage of channels that can be considered hubs
    num_element = length(b_matrix);
    
    %% Caculate the unweighted degree of the network
    channels_degree = degrees_und(b_matrix);

    %% Threshold the degree to keep only t
    sorted_degree = sort(channels_degree);
    t_index = floor(num_element*(1 - t_level)) + 1;
    t_element = sorted_degree(t_index);    
    
    %% Threshold the degrees
    t_channels_degree = channels_degree;
    t_channels_degree(t_channels_degree < t_element) = 0;
    
    % Get only the hub degree
    non_zero_hub_degree = [];
    for i = 1:num_element
        current_degree = t_channels_degree(i);
        if(current_degree > 0)
            non_zero_hub_degree(end+1) = current_degree; 
        end
    end
    
    % get the median value of the non-zero hub degree
    % to detect which region should be returned (we don't look at extreme
    % values)
    m_hub_degree = find_hub(non_zero_hub_degree);
    disp("The degree of the selected hub:");
    disp(m_hub_degree)
    m_hub_degree_index = find(channels_degree == m_hub_degree);  
    median_degree = threshold_anterior_posterior(m_hub_degree_index, location);
end


% This will try to find the middle degree value not looking at extreme
% values
function [middle_value] = find_hub(hubs)
    sorted_hub_degree = sort(hubs);
    middline_index = floor(length(sorted_hub_degree)/2);
    middle_value = sorted_hub_degree(middline_index);
end

function [normalized_value] = threshold_anterior_posterior(index,channels_location)
%THRESHOLD_ANTERIOR_POSTERIOR Summary of this function goes here
%   Detailed explanation goes here

    current_x = channels_location(index).X;
    
    all_x = zeros(1,length(channels_location));
    for i = 1:length(channels_location)
       all_x(i) = channels_location(i).X; 
    end
    
    min_x = min(all_x);
    max_x = max(all_x);
    
    normalized_value = (current_x - min_x)/(max_x - min_x);
end