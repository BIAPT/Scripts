%{
    Yacine Mahdid 2019-11-01
    This script is to test the generate_graph_feature_vector function
    (which should produce same behavior as in experiment 1)
%}

%% Variables Initialization
num_regions = 82; % Number of source localized regions
num_null_network = 10; % Number of null network to create 
bin_swaps = 10;  % used to create the null network
weight_frequency = 0.1; % used to create the null network
t_level = 0.1; % Threshold level (keep 10%)

%% Generating fake matrix for testing purpose
fake_graph = rand(num_regions, num_regions);

X = generate_graph_feature_vector(fake_graph, num_null_network, bin_swaps, weight_frequency, t_level);