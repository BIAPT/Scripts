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
