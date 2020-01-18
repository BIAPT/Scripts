%{
    This script was written by Yacine Mahdid 2019-12-09
    It's purpose it to investigate the old and new dpli algorithm
%}

old_data_location = 'C:\Users\biapt\Desktop\motif fix\dpli old\old_ec1_dpli_alpha.mat';

data = load(old_data_location);
old_dpli = data.z_score;

%% Variables Initalization
data_location = 'C:\Users\biapt\Desktop\motif fix\mdfa17_data';
participant = 'MDFA17';
epochs = {'BASELINE'};

% Experiment variables
% for dPLI
frequency_band = [8 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;

filename = strcat(participant,'_',epochs{1},'.set');
recording = load_set(filename,data_location);
    
% Calculate the dPLI
result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

new_dpli = result_dpli.data.avg_dpli;

plot_wpli(old_dpli, "Old dPLI", "", "jet", 0)
plot_wpli(new_dpli, "New dPLI", "", "jet", 0)

% for Motif Analysis
number_rand_network = 10;
bin_swaps = 10;
weight_frequency = 0.1;

% make a phase lead matrix using the average dPLI
network = make_phase_lead(new_dpli);

% filtering the non-scalp channels
channels_location = result_dpli.metadata.channels_location;
[network,channels_location] = filter_non_scalp(network,channels_location);

% Calculate the frequency/source/target/distance of each motifs
[intensity, coherence, frequency, source, target, distance] = motif_3(network, channels_location, number_rand_network, bin_swaps, weight_frequency);

 % Generate all figures for this particular participant
plot_motif(frequency, source, target, distance,channels_location, "test",1);
plot_motif(frequency, source, target, distance,channels_location,"TEST",7);

function plot_motif(frequency, source, target, distance,channels_location, title_name, motif_index)

    % Get the Z-Score of each values to plot in the topographic map
    norm_frequency = normalize_motif(frequency);
    norm_source = normalize_motif(source);
    norm_target = normalize_motif(target);
    norm_distance = normalize_motif(distance);
    
    % Will plot motif 
    figure;
    ax1 = subplot(2,2,1);
    title(strcat("Frequency for Motif ",string(motif_index)," ", title_name));
    topoplot(norm_frequency(motif_index,:),channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    ax2 = subplot(2,2,2);
    title(strcat("Distance for Motif ",string(motif_index)," ", title_name));
    topoplot(norm_distance(motif_index,:),channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    ax3 = subplot(2,2,3);
    title(strcat("Source for Motif ",string(motif_index)," ", title_name));
    topoplot(norm_source(motif_index,:),channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    ax4 = subplot(2,2,4);
    title(strcat("Target for Motif ",string(motif_index)," ",title_name));
    topoplot(norm_target(motif_index,:),channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    
    colormap(ax1,'jet');
    colormap(ax2,'bone');
    colormap(ax3,'hot');
    colormap(ax4,'winter');
end
