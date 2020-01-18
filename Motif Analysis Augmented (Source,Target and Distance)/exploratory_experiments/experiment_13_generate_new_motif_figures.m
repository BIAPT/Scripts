%{
    Script written by Yacine Mahdid
    The goal of this script is to load the right motifs and to generate the
    figures of distance source and frequency
%}

%% Variables Initalization
data_location = 'E:\research projects\motif_analysis\motif\';
output_location = 'E:\research projects\motif_analysis\figure\';

% 03 05 06 11 12 15
participant = 'MDFA15';

epochs = {'BASELINE', 'IF5', 'EMF5','EML30','EML10','EML5','RECOVERY'};
frequency = 'alpha';

output_location = strcat(output_location,filesep);

data = load(strcat(data_location,participant,'.mat'));
state_data = data.state_data;
data  = load(strcat(data_location,participant,'_eeg_info.mat'));
channels_location = data.EEG_info.chanlocs;

%% Iterating over the files
for e_i = 1:length(epochs)
    % Load the data
    disp(strcat("Generate motifs figures from ", participant," at ", epochs{e_i}));
    
    motifs = state_data(e_i);
    
    fake_network = zeros(length(channels_location), length(channels_location));
    [fake_network, channels_location] = filter_non_scalp(fake_network, channels_location);

    % Calculate the frequency/source/target/distance of each motifs
    plot_motif(motifs.frequency, motifs.source, motifs.distance,channels_location, strcat(participant,' at ',epochs{e_i}),1);
    filename = strcat(output_location,participant,'_M1_',epochs{e_i},'.fig');
    savefig(filename)
    close(gcf)
    plot_motif(motifs.frequency, motifs.source, motifs.distance,channels_location, strcat(participant,' at ',epochs{e_i}),7);
    filename = strcat(output_location,participant,'_M7_',epochs{e_i},'.fig');
    savefig(filename)
    close(gcf)
end

function [handle] = plot_motif(frequency, source, distance, channels_location, title_name, motif_index)

    % Get the Z-Score of each values to plot in the topographic map
    norm_frequency = normalize_motif(frequency);
    norm_source = normalize_motif(source);
    norm_distance = normalize_motif(distance); %normalize_motif(distance);
    
    % Will plot motif 
    figure;
    ax1 = subplot(1,3,1);
    title(strcat("Frequency for Motif ",string(motif_index)," ", title_name));
    topoplot(norm_frequency(motif_index,:),channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    ax2 = subplot(1,3,2);
    title(strcat("Distance for Motif ",string(motif_index)," ", title_name));
    topoplot(norm_distance(motif_index,:),channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    ax3 = subplot(1,3,3);
    title(strcat("Source for Motif ",string(motif_index)," ", title_name));
    topoplot(norm_source(motif_index,:),channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    
    colormap(ax1,'jet');
    colormap(ax2,'bone');
    colormap(ax3,'hot');
end

function  [motifs,channels_location] = filter_mdfa17(motifs, channels_location)
%FILTER_NON_SCALP Summary of this function goes here
%   Detailed explanation goes here
    non_scalp_channel_label = {'E8', 'E21', 'E25', 'E81', 'E88'};

    for i=1:length(non_scalp_channel_label)
        current_label = non_scalp_channel_label{i};
        for j=1:length(channels_location)
           if(strcmp(channels_location(j).labels,current_label))
               channels_location(j) = [];
               motifs.frequency(:,j) = [];
               motifs.source(:,j) = [];
               motifs.target(:,j) = [];
               motifs.distance(:,j) = [];
               break;
           end
        end
    end
end


