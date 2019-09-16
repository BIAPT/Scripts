function [similarities] = matrix_cosine_similarity(A,B)
%   MATRIX_COSINE_SIMILARITY  measures similarity between two non-zero
%   matrices.
%   A: is non-zero matrix of size L*N
%   B: is non-zero matrix of size M*N
%
%   similarities: is a matrix of size L*M which represent the similarity of all dimension (L and M) of A and B
%   ranging from [-1,1]
    
    %% Variable Initalization
    number_row_a = size(A,1);
    number_row_b = size(B,1);
    number_col_a = size(A,2);
    number_col_b = size(B,2);

    similarities = zeros(number_row_a,number_row_b);
    
    %% Error checking
    % Check if we have the same number of columns
    if(number_col_a ~= number_col_b)
       error("The matrix A and the matrix B need tbe of same lenght.");
    end
    
    % It is allowed to have different number of rows as we will get a
    % similarities matrix of length_row_a*length_row_b
    
    %% Iterate over the row and calculate vector cosine similarities
    for row_a_i = 1:number_row_a
        a = A(row_a_i);
        for row_b_i = 1:number_row_b
            b = B(row_b_i);
            similarity = vector_cosine_similarity(a,b);
            similarities(row_a_i,row_b_i) = similarity;
        end
    end
    
end
