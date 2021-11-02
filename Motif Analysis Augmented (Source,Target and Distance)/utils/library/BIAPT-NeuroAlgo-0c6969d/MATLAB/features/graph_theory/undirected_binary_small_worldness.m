function [b_small_worldness] = undirected_binary_small_worldness(b_matrix,null_networks)
%BINARY_SMALL_WORLDNESS calculate the ration of cc and ge
%   b_matrix: a N*N binary square matrix
%   null_networks: 3d matrix containing pre-made null_networks

    %% Calculate Clustering Coefficient and Charpath (avg path length)
    [~, norm_c_coeff] = undirected_binary_clustering_coefficient(b_matrix,null_networks);
    [~,~,~,norm_avg_path_length] = binary_global_efficiency(b_matrix,null_networks);
    
    %% Calculate binary small worldness on the normalized features
    b_small_worldness = norm_c_coeff/norm_avg_path_length; % binary smallworldness
   
end

