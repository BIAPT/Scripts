function [c_coeff, norm_average_c_coeff] = undirected_weighted_clustering_coefficient(matrix,null_networks)
%CLUSTERING_COEFFICIENT Will calculate the clusterig coefficient for th
%binary matrix
%   matrix: a N*N weighted square matrix
%   null_networks: 3d matrix containing pre-made null_networks
                
    %% Find Clustering coefficient
    
    norm_matrix = normalize_matrix(matrix);
    c_coeff = clustering_coef_wu(norm_matrix);  
    
    %% Calculate the characteristic path length for each null_matrix and average them
    [number_null_network, ~, ~] = size(null_networks);
    null_network_c_coeff = zeros(length(c_coeff),number_null_network);
    for i = 1:number_null_network
        null_w_matrix = squeeze(null_networks(i,:,:));
        norm_null_matrix = normalize_matrix(null_w_matrix);
        null_network_c_coeff(:,i) = clustering_coef_wu(norm_null_matrix);
    end
    avg_null_network_c_coeff = mean(null_network_c_coeff,2);
    
    c_coeff = c_coeff / nanmean(avg_null_network_c_coeff); % normalize the clustering coefficient of every channel too
    norm_average_c_coeff = nanmean(c_coeff)/nanmean(avg_null_network_c_coeff); % weighted clustering coefficient
end

function [norm_matrix] = normalize_matrix(matrix)
    %matrix = (matrix - min(matrix(:))) ./ (max(matrix(:)) - min(matrix(:)));
    %norm_matrix = weight_conversion(matrix, 'normalize');
    norm_matrix = abs(matrix);
end