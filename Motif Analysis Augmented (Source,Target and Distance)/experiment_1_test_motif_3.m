%{
    Script written by Yacine Mahdid at 2019-12-01 for the review of the
    motif analysis paper. 
    The goal is to get the source, target and the average distance of the
    motif connection.
%}

%% Using real data to test the motif_3.m script inside the NA library

% Load the data whatever it is
%{
data_folder = '/test_data'; % where the data is
filename = 'test_data.set'; % which data you want to run this analysis on
[filepath,name,ext] = fileparts(mfilename('fullpath'));
test_data_path = strcat(filepath,data_folder);
recording = load_set(filename,test_data_path);

% Variables initialization for dPLI calculation
% dPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
%}
% Variables initialization for motif analysis
network = make_phase_lead(result_dpli.data.avg_dpli);
number_rand_network = 10;
bin_swaps = 10;
weight_frequency = 0.1;
channels_location = result_dpli.metadata.channels_location;
[intensity, coherence, frequency, source, target, distance] = motif_3(network, channels_location, number_rand_network, bin_swaps, weight_frequency);

% Here we will normalize the output of this (only the frequency) and plot
% it

% We normalize by using the z score
[norm_frequency] = normalize_motif(frequency);
plot_motif(norm_frequency(7,:),"Motif 7 at alpha",result_dpli.metadata.channels_location,'summer');

% Source and target
norm_source = normalize_motif(source);
norm_target = normalize_motif(target);
plot_motif(norm_source(7,:),"Source Location of Motif 7 at Alpha ",result_dpli.metadata.channels_location,'hot');
plot_motif(norm_target(7,:),"Target Location of Motif 7 at Alpha ",result_dpli.metadata.channels_location,'winter');

% distances
norm_distance = normalize_motif(distance);
plot_motif(norm_distance(7,:), "Connection Distance of Motif 7 at Alpha", result_dpli.metadata.channels_location,'bone');