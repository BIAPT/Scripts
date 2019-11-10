function [X] = generate_weighted_graph_feature_vector(graph, num_null_network, bin_swaps, weight_frequency, transform)
%GENERATE_FEATURE_VECTOR calculate graph theory feature
%   This is building on the experiment using binary graph classification
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
    
    % Generate the null networks
    null_networks = generate_null_networks(graph, num_null_network, bin_swaps, weight_frequency);

    %% Calculate each of the weighted graph theory metric
    % Weighted Clustering Coefficient
    % Here we are using the log transform, however I'm not sure if I need
    % to use the inverse distance
    [~,norm_g_eff,~,~] = weighted_global_efficiency(graph, null_networks, transform);

    % Modularity
    community = modularity(graph);

    % Weighted Smallworldness
    w_small_worldness = undirected_weighted_small_worldness(graph,null_networks,transform);

    % Binary Clustering Coefficient
    [clust_coeff, norm_avg_clust_coeff] = undirected_weighted_clustering_coefficient(graph,null_networks);
    
    %% Features vector construction
    X = [mean_graph; std_graph; clust_coeff; norm_avg_clust_coeff; norm_g_eff; community; w_small_worldness];
end

