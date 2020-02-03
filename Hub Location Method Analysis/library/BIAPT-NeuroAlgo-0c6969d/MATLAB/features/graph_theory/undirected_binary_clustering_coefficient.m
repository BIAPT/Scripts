function [c_coeff, norm_average_c_coeff] = undirected_binary_clustering_coefficient(b_matrix,null_networks)
%CLUSTERING_COEFFICIENT Will calculate the clusterig coefficient for th
%binary matrix
%   b_matrix: a N*N binary square matrix
%   null_networks: 3d matrix containing pre-made null_networks
                
    %% Find Clustering coefficient
    c_coeff = clustering_coef_bu(b_matrix);  
    
    %% Calculate the characteristic path length for each null_matrix and average them
    [number_null_network, ~, ~] = size(null_networks);
    null_network_c_coeff = zeros(length(c_coeff),number_null_network);
    for i = 1:number_null_network
        null_b_matrix = squeeze(null_networks(i,:,:));
        null_network_c_coeff(:,i) = clustering_coef_bu(null_b_matrix);
    end
    avg_null_network_c_coeff = mean(null_network_c_coeff,2);
    
    norm_average_c_coeff = nanmean(c_coeff)/nanmean(avg_null_network_c_coeff); % binary clustering coefficient
end

