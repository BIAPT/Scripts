function [normalized_matrix] = normalize_data(matrix)
%NORMALIZE_DATA Summary of this function goes here
%   Detailed explanation goes here

    epsilon = 0.000001; % to avoid a divide by zero if the std is 0
    mean_matrix = mean(matrix);
    std_matrix = std(matrix);
    
    normalized_matrix = (matrix - min(matrix)) ./ (max(matrix) - min(matrix));
end

