function [b_small_worldness] = undirected_weighted_small_worldness(matrix,null_networks, transform)
%BINARY_SMALL_WORLDNESS calculate the ration of cc and ge
%   matrix: a N*N weighted square matrix
%   null_networks: 3d matrix containing pre-made null_networks

    %% Calculate Clustering Coefficient and Charpath (avg path length)
    [~, norm_c_coeff] = undirected_weighted_clustering_coefficient(matrix,null_networks);
    [~,~,~,norm_avg_path_length] = weighted_global_efficiency(matrix,null_networks, transform);
    
    %% Calculate binary small worldness on the normalized features
    b_small_worldness = norm_c_coeff/norm_avg_path_length; % binary smallworldness
   
end

