function normalized_matrix = square_normalization(matrix)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    normalized_matrix = (matrix - min(matrix)) ./ (max(matrix) - min(matrix)); % This gives us result bounded by [0 and 1]
    normalized_matrix = abs(1 - normalized_matrix); % This give us normalized proximity
    
    symmetric_normalized_matrix = normalized_matrix;
    % Restore the symmetry and make sure the row and column sum to 1
    for i = 1:length(matrix)
       for j = 1:length(matrix) 
          symmetric_normalized_matrix(i,j) = normalized_matrix(i,j);
          symmetric_normalized_matrix(j,i) = symmetric_normalized_matrix(i,j); 
       end
    end
    
    normalized_matrix = symmetric_normalized_matrix;
end

