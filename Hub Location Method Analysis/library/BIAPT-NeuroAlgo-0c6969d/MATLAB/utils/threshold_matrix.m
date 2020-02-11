function [t_matrix] = threshold_matrix(matrix,t_level)
%THRESHOLD_MATRIX Threshold a matrix to have element below a significant
%amount to 0
%   Matrix: N*N matrix with value within any range
%   t_level: value from 0 to 1 which set the ratio of highest value
%   to keep. i.e. t_level = 0.05 -> keep only top 5% value
    
    %% Flatten the matrix into an array
    [num_row, num_col] = size(matrix);
    num_element = num_row*num_col;
    array = reshape(matrix, [1 num_element]);
    
    %% Find the value to threshold on (the one at the limit of top t_level%
    sorted_array = sort(array);
    t_index = floor(num_element*(1 - t_level)) + 1;
    t_element = sorted_array(t_index);
    
    %% Threshold the matrix
    t_matrix = matrix;
    t_matrix(t_matrix < t_element) = 0;
    
    %% Remove the diagonal elements
    t_matrix = t_matrix - diag(diag(t_matrix));
end

