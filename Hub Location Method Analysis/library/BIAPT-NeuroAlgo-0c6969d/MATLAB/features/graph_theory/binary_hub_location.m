function [normalized_location,previous_location, channels_degree] = binary_hub_location(b_matrix, channels_location, t_level)
%HUB_LOCATION choose from the the channels the channel with the highest
%degree
% Input:
%   b_matrix: a binary undirected matrix
%   t_level: what top percentage represent a hub in the brain (what is the
%   definition of a hub)
     
    num_element = length(b_matrix);
    
    
    %% Caculate the unweighted degree of the network
    channels_degree = degrees_und(b_matrix);
    
    %% Calculate previous location
    [~, channel_index] = max(channels_degree);
    previous_location = threshold_anterior_posterior(channel_index, channels_location);

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
    disp("Non zero hub degree:")
    non_zero_hub_degree
    % get the median value of the non-zero hub degree
    % to detect which region should be returned (we don't look at extreme
    % values)
    m_hub_degree = find_hub(non_zero_hub_degree);
    disp("The degree of the selected hub:");
    disp(m_hub_degree)
    m_hub_degree_index = find(channels_degree == m_hub_degree);  
    normalized_location = threshold_anterior_posterior(m_hub_degree_index, channels_location);
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

