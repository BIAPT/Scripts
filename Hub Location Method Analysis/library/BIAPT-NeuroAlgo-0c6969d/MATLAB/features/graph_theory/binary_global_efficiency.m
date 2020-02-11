function [g_efficiency,norm_g_efficiency,avg_path_length,norm_avg_path_length] = binary_global_efficiency(b_matrix,null_networks)
%GLOBAL EFFICIENCY will calculate the global
%efficiency and path length for connectivity matrices like wpli
%   b_matrix: a N*N binary square matrix
%   null_networks: 3d matrix containing pre-made null_networks
    
    %% Calculate the characteristic path length
    input_distance = distance_bin(b_matrix);
    [avg_path_length,g_efficiency,~,~,~] = charpath(input_distance,0,0);
    
    %% Calculate the characteristic path length for each null_matrix and average them
    [number_null_network, ~, ~] = size(null_networks);
    null_network_avg_path_length = zeros(1,number_null_network);
    null_network_g_efficiency = zeros(1,number_null_network);
    for i = 1:number_null_network
        null_b_matrix = squeeze(null_networks(i,:,:));
        null_input_distance = distance_bin(null_b_matrix);
        [null_matrix_avg_path_length,null_matrix_g_efficiency,~,~,~] = charpath(null_input_distance,0,0);   % binary charpath    
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

