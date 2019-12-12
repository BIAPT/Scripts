%{
    Script written by Yacine Mahdid
    The goal of this script is to load the right dPLI matrices and then to
    calculate the motifs along with the augmented values.

    The choosen participant was MDFA17 as proposed by Danielle Nadin
    The timepoints are:
    baseline, induction, unconscious, - 30min, -10min, -5min, +30min,
    +180min
%}

%% Variables Initalization
data_location = 'C:\Users\biapt\Desktop\motif fix\mdfa17_dpli_data\';
output_location = 'C:\Users\biapt\Desktop\motif fix\mdfa17_motif_data\';
participant = 'MDFA17';
epochs = {'BASELINE', 'IF5', 'EMF5', 'EML30', 'EML10', 'EML5', 'RECOVERY'};

% Experiment variables
% for Motif Analysis
number_rand_network = 100;
bin_swaps = 10;
weight_frequency = 0.1;


%% Iterating over the files
for e_i = 1:length(epochs)
    % Load the data
    
    filename = strcat(data_location,participant,'_',epochs{e_i},'_dpli.mat');
    disp(strcat("Motif analysis on dpli value from ", participant," at ", epochs{e_i}));
    data = load(filename);
    
    % Get the data out
    result_dpli = data.result_dpli;
    channels_location = result_dpli.metadata.channels_location;
    
    % make a phase lead matrix using the average dPLI
    network = make_phase_lead(result_dpli.data.avg_dpli);
    
    [network,channels_location] = filter_non_scalp(network,channels_location);
    
    % Calculate the frequency/source/target/distance of each motifs
    motifs = struct();
    [motifs.frequency, motifs.source, motifs.target, motifs.distance] = motif_3(network, channels_location, number_rand_network, bin_swaps, weight_frequency);
    
    % Generate all figures for this particular participant
    plot_motif(motifs.frequency, motifs.source, motifs.target, motifs.distance,channels_location, strcat(participant,' at ',epochs{e_i}),1);
    plot_motif(motifs.frequency, motifs.source, motifs.target, motifs.distance,channels_location, strcat(participant,' at ',epochs{e_i}),7);

    
    output_filename = strcat(output_location,participant,'_',epochs{e_i},'_motif.mat');
    save(output_filename,'motifs');
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




