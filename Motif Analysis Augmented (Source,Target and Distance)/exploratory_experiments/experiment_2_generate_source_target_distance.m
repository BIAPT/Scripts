%{
    Script written by Yacine Mahdid
    The goal of this script is to load the right dataset generate the right
    dPLI matrices and generate the frequency/source/target/distance plots
    for one participant at every time points.

    The choosen participant was MDFA17 as proposed by Danielle Nadin
    The timepoints are:
    baseline, induction, unconscious, - 30min, -10min, -5min, +30min,
    +180min
%}

%% Variables Initalization
data_location = 'C:\Users\biapt\Desktop\motif fix\mdfa17_data';
participant = 'MDFA17';
epochs = {'BASELINE', 'IF5', 'EMF5', 'EML30', 'EML10', 'EML5', 'RECOVERY'};

% Experiment variables
% for dPLI
frequency_band = [8 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;

% for Motif Analysis
number_rand_network = 10;
bin_swaps = 10;
weight_frequency = 0.1;

%% Iterating over the files
for e_i = 1:length(epochs)
    % Load the data
    filename = strcat(participant,'_',epochs{e_i},'.set');
    recording = load_set(filename,data_location);
    % Calculate the dPLI
    result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
    % make a phase lead matrix using the average dPLI
    network = make_phase_lead(result_dpli.data.avg_dpli);
    
    % filtering the non-scalp channels
    channels_location = result_dpli.metadata.channels_location;
    [network,channels_location] = filter_non_scalp(network,channels_location);
    
    % Calculate the frequency/source/target/distance of each motifs
    [intensity, coherence, frequency, source, target, distance] = motif_3(network, channels_location, number_rand_network, bin_swaps, weight_frequency);

    % Generate all figures for this particular participant
    plot_motif(frequency, source, target, distance,channels_location, strcat(participant,' at ',epochs{e_i}),1);
    plot_motif(frequency, source, target, distance,channels_location, strcat(participant,' at ',epochs{e_i}),7);
end

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


