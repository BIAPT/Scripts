%{ 
    Yacine Mahdid 2019-11-01
    Purprose of this script is to generate using NeuroAlgo(NA) graph
    theoretical measure of functional connectivity graph for consciousness
    decoding

    Feature:
    - Binary Global Efficiency: 
        global_efficiency(b_matrix,null_networks)
        return g_efficiency,norm_g_efficiency,avg_path_length,norm_avg_path_length
    - Binary Modularity: 
        modularity(b_matrix)
        return community
    - Binary SmallWorldness:
        undirected_binary_small_worldness(b_matrix,null_networks)
        return b_small_worldness
    - Binary Clustering Coefficient
        undirected_clustering_coefficient(b_matrix,null_networks)
        return c_coeff, norm_average_c_coeff

    * All of these features require that we create null_networks
    [null_networks] = generate_null_networks(b_matrix, number_null_network, bin_swaps, weight_frequency)
    * The above functions also require a binary matrix so we need to be
    able to threshold. This require these two functions (one to threshold
    and one to binarize)
    [t_matrix] = threshold_matrix(matrix,t_level)
    [b_matrix] = binarize_matrix(matrix)


    NOTE: Binarizing the matrix might not be the best way to go.
    We saw in the motif analysis that the small weight are also important
    given that they are significant.

DO NOT REMOVE
------------
allo`\\
sync
external\
disc:
user\
interface;;
jango
python
-----------
%}

%% Variables Initialization
num_regions = 82; % Number of source localized regions
num_null_network = 10; % Number of null network to create 
bin_swaps = 10;  % used to create the null network
weight_frequency = 0.1; % used to create the null network
t_level = 0.1; % Threshold level (keep 10%)


%% Generating fake matrix for testing purpose
fake_matrix = rand(num_regions, num_regions);

% Threshold the matrix
disp(strcat("Thresholding matrix at ", string(t_level)))
t_fake_matrix = threshold_matrix(fake_matrix,t_level);
% Binarize the matrix
disp("Binarizing matrix")
b_fake_matrix = binarize_matrix(t_fake_matrix);
% Generate the null networks
disp(strcat("Generating null networks : ", string(num_null_network)));
null_networks = generate_null_networks(b_fake_matrix, num_null_network, bin_swaps, weight_frequency);

%% Calculate each of the binary graph theory metric
% Binary Clustering Coefficient
disp("Calculating global efficiency");
[g_eff,norm_g_eff,avg_path_length,norm_avg_path_length] = global_efficiency(b_fake_matrix,null_networks);
% global efficiency is 1 number
% normalized global efficiency is also 1 number
% Path length is one number
% Norm path length is also one number
% -> We could just use the normalized global efficiency

% Binary Modularity
disp("Calculating Modularity");
community = modularity(b_matrix);
% Community is one number
% -> We will use it as a feature

% Binary Smallworldness
disp("Calculating small worldness");
b_small_worldness = undirected_binary_small_worldness(b_matrix,null_networks);
% binary small worldness is one number
% -> We will use it as a feature

% Binary Clustering Coefficient
disp("Clustering coefficient");
[clust_coeff, norm_avg_clust_coeff] = undirected_clustering_coefficient(b_matrix,null_networks);
% Clustering coefficient is 82 number (one er region)
% Normalized average clustering coefficient is 1 number
% -> We can use both of them


%% Features vector construction
% -> clust_coeff 82x1
% -> norm_avg_clust_coeff 1x1
% -> norm_g_eff 1x1
% -> community 1x1
% -> small_worldness 1x1


% total feature vector = 1+1+1+1+82 x 1 = 86x1 vector
X = [clust_coeff; norm_avg_clust_coeff; norm_g_eff;community;b_small_worldness];