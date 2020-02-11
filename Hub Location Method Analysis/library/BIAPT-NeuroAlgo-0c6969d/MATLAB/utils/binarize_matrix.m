function [b_matrix] = binarize_matrix(matrix)
%BINARIZE_MATRIX set the value of the matrix to 0 or 1
%   matrix: a N*N matrix
    b_matrix = matrix;
    b_matrix(b_matrix > 0) = 1;
end

