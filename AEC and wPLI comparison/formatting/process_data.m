function [processed_data] = process_data(data,regions_mask)
%PROCESS_DATA Summary of this function goes here
%   Detailed explanation goes here
    filtered_data = filter_regions(data,regions_mask);
    averaged_data = mean(filtered_data,3);
    std_data = std(filtered_data,0,3);
    processed_data = [averaged_data std_data];
end

