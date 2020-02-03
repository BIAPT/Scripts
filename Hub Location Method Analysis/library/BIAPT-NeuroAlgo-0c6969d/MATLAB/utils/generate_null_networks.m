function [null_networks] = generate_null_networks(matrix, number_null_network, bin_swaps, weight_frequency)
%GENERATE_NULL_NETWORK Summary of this function goes here
%   Detailed explanation goes here
    
    %% Create random null network
    [num_row, num_col] = size(matrix); % These should be the same
    null_networks = zeros(number_null_network,num_row, num_col);
    
    % Here we use multi-core to speed up the analysis (this is a
    % bottleneck)
    is_binary = is_matrix_binary(matrix);
    parfor i = 1:number_null_network
        if(is_binary)
            [null_matrix,~] = randmio_und(matrix,bin_swaps);  
        else
            [null_matrix,~] = null_model_und_sign(matrix,bin_swaps,weight_frequency);  
        end
        null_networks(i,:,:) = null_matrix; % store all null matrix
    end

end

function result = is_matrix_binary(matrix)
    result = 1;
    for i = 1:length(matrix)
        
        for j =1:length(matrix)
            if(matrix(i,j) ~= 0 && matrix(i,j) ~= 1)
               result = 0;
               return;
            end
        end
    end
    
end