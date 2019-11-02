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
    * THe

allo`\\
sync
external\
disc:
user\
interface;;
jango
python

    
%}

%% Variables Initialization
num_regions = 82;


%% Generating fake matrix for testing purpose
fake_data = rand(num_regions, num_regions);


