function [similarities] = matrix_cosine_similarity(A,B)
%   MATRIX_COSINE_SIMILARITY  measures similarity between two non-zero
%   matrices.
%   A is non-zero matrix of size L*N
%   B is non-zero matrix of size M*N
%
%   similarities is a matrix of size L*M which represent the similarity of all dimension (L and M) of A and B
%   ranging from [0,1]

    
    %% Error checking
    % Check if we have the same number of columns
    if(size(A,2) ~= size(B,2))
       error("The matrix A and the matrix B need tbe of same lenght.");
    end
    
    % It is allowed to have different number of rows as we will get a
    % similarities matrix of length_row_a*length_row_b
    
    %% Iterate over one dimension
    dimension_a = size(A,1);
    dimension_b = size(B,1);
    similarities = zeros(dimension_a,dimension_b);
    
    for i = 1:dimension_a
        a = A(i);
        for j = 1:dimension_b
            b = B(j);
            similarity = vector_cosine_similarity(a,b);
            similarities(i,j) = similarity;
        end
    end
    
end
