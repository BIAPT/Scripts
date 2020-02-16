function [g_efficiency, norm_g_efficiency, avg_path_length, norm_avg_path_length] = weighted_global_efficiency(matrix, null_networks, transform)
%GLOBAL EFFICIENCY will calculate the global
%efficiency and path length for connectivity matrices like wpli
%   matrix: a N*N weighted square matrix
%   null_networks: 3d matrix containing pre-made null_networks
%   transform: either 'log' or 'inv' (see distance_wei_floy)
    
    %% Calculate the characteristic path length
    norm_matrix = normalize_matrix(matrix);
    input_distance = distance_wei_floyd(norm_matrix,transform); 
    [avg_path_length,g_efficiency,~,~,~] = charpath(input_distance,0,0); 
    
    %% Calculate the characteristic path length for each null_matrix and average them
    [number_null_network, ~, ~] = size(null_networks);
    null_network_avg_path_length = zeros(1,number_null_network);
    null_network_g_efficiency = zeros(1,number_null_network);
    
    for i = 1:number_null_network
        null_w_matrix = squeeze(null_networks(i,:,:));
        
        norm_null_matrix = normalize_matrix(null_w_matrix);
        
        null_input_distance = distance_wei_floyd(norm_null_matrix, transform);
        [null_matrix_avg_path_length,null_matrix_g_efficiency,~,~,~] = charpath(null_input_distance,0,0);  
        
        null_network_avg_path_length(i) = null_matrix_avg_path_length;
        null_network_g_efficiency(i) = null_matrix_g_efficiency;
    end
    
    % Calculate mean null path length and mean global efficiency
    null_avg_path_length = mean(null_network_avg_path_length);
    null_g_efficiency = mean(null_network_g_efficiency);
    
    %% Normalizing average path length and global effiency
    norm_g_efficiency = g_efficiency/null_g_efficiency;
    norm_avg_path_length = avg_path_length/null_avg_path_length;
end


function [norm_matrix] = normalize_matrix(matrix)
    %matrix = (matrix - min(matrix(:))) ./ (max(matrix(:)) - min(matrix(:)));
    %norm_matrix = weight_conversion(matrix, 'normalize');
    norm_matrix = abs(matrix);
end