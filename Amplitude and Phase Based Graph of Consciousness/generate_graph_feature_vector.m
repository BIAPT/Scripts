function [X] = generate_graph_feature_vector(graph, num_null_network, bin_swaps, weight_frequency, t_level)
%GENERATE_FEATURE_VECTOR calculate graph theory feature
%   This is based on experiment_1 and will calculate the following feature
%   vector:
% -> mean 82x1
% -> std 82x1
% -> clust_coeff 82x1
% -> norm_avg_clust_coeff 1x1
% -> norm_g_eff 1x1
% -> community 1x1
% -> small_worldness 1x1
% X is a 86x1 feature vector and the first 82 map to the source localized
% regions
%
% graph here is a functional connectivity matrix

    % Calculate the unbinarized features
    % Mean 
    mean_graph = mean(graph,2);
    std_graph = std(graph,0,2);
    
    % Threshold the matrix
    t_grap = threshold_matrix(graph,t_level);
    % Binarize the matrix
    b_graph = binarize_matrix(t_grap);
    % Generate the null networks
    null_networks = generate_null_networks(b_graph, num_null_network, bin_swaps, weight_frequency);

    %% Calculate each of the binary graph theory metric
    % Binary Clustering Coefficient
    [~,norm_g_eff,~,~] = global_efficiency(b_graph,null_networks);

    % Binary Modularity
    community = modularity(b_graph);

    % Binary Smallworldness
    b_small_worldness = undirected_binary_small_worldness(b_graph,null_networks);

    % Binary Clustering Coefficient
    [clust_coeff, norm_avg_clust_coeff] = undirected_clustering_coefficient(b_graph,null_networks);
    
    %% Features vector construction
    X = [mean_graph;std_graph;clust_coeff; norm_avg_clust_coeff; norm_g_eff;community;b_small_worldness];
end

