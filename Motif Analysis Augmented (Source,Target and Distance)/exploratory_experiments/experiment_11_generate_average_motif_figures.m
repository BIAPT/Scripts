%{
    Script written by Yacine Mahdid
    The goal of this script is to load the right motifs and to generate the
    figures for the average participants motif.
%}

%% Variables Initalization
data_location = 'C:\Users\biapt\Desktop\motif_analysis_dst\motif\';
output_location = 'C:\Users\biapt\Desktop\motif_analysis_dst\figure\';

participants = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17'};
epochs = {'EC1', 'IF5', 'EF5', 'EL30', 'EL10', 'EL5', 'EC8'};

output_location = strcat(output_location,"AVG",filesep);

%% First step is to find the smallest set of channels to calculate the average on
avg_channels_location = [];

%% Iterating over the files
avg_motifs = struct();
avg_motifs.frequency = zeros(7,13,99);
avg_motifs.source = zeros(7,13,99);
avg_motifs.distance = zeros(7,13,99);

% Here we will iterate over all the participant load them and then add them
% together if their frequency at a given motif is not zero.


for p_i = 1:length(participants)
    participant = participants{p_i};
    for e_i = 1:length(epochs)
        % Load the data
        disp(strcat("Aggregating from ", participant," at ", epochs{e_i}));

        base = strcat(data_location,participant,filesep);
        filename = strcat(base,participant,'_',epochs{e_i},'_motif.mat');
        data = load(filename);
        motifs = data.motifs;

        filename = strcat(base,'eeg_info.mat');
        data = load(filename);
        channels_location = data.EEG_info.chanlocs;
        
        if(p_i == 1 && length(channels_location) == 129 && length(avg_channels_location) < 99)
            avg_channels_location = filter_non_scalp(channels_location);
        end
        
        % Here we make sure that both the average and the current motifs
        % have same size
        [channels_location] = filter_non_scalp(channels_location);
        
        [avg_motifs, motifs_count] = add_motifs(avg_motifs, avg_channels_location,e_i, motifs, channels_location, motifs_count);
        
    end
end

% do the average [ TODO ]

for e_i = 1:length(epochs)
    for m_i = 1:13
        if(motifs_count(e_i,m_i) ~= 0)
            for c_i = 1:99
                avg_motifs.frequency(e_i,m_i,c_i) = avg_motifs.frequency(e_i,m_i,c_i) / motifs_count(e_i,m_i,c_i);
                avg_motifs.source(e_i,m_i,c_i) = avg_motifs.source(e_i,m_i,c_i) / motifs_count(e_i,m_i,c_i);
                avg_motifs.distance(e_i,m_i,c_i) = avg_motifs.distance(e_i,m_i,c_i) / motifs_count(e_i,m_i,c_i);   
            end
        end
    end
end


% at this spot we should have our averaged motifs structure

for e_i = 1:length(epochs)
    frequency = squeeze(avg_motifs.frequency(e_i,:,:));
    source = squeeze(avg_motifs.source(e_i,:,:));
    distance = squeeze(avg_motifs.distance(e_i,:,:));
   
   
    title_name = strcat("Average participant at ", epochs{e_i});
    plot_motif(frequency, source, distance, avg_channels_location, strcat(title_name," for motif 1"), 1)
    filename = strcat(output_location,'AVG_M1_',epochs{e_i},'.fig');
    savefig(filename)
    close(gcf)
    
    plot_motif(frequency, source, distance, avg_channels_location, strcat(title_name," for motif 7"), 7)
    filename = strcat(output_location,'AVG_M7_',epochs{e_i},'.fig');
    savefig(filename)
    close(gcf)
end

function plot_motif(frequency, source, distance, channels_location, title_name, motif_index)

    % Get the Z-Score of each values to plot in the topographic map
    norm_frequency = normalize_motif(frequency);
    norm_source = normalize_motif(source);
    norm_distance = distance; %normalize_motif(distance);
    
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

function  [avg_channels_location] = fit_channels_location(avg_channels_location, channels_location)
%FILTER_NON_SCALP Summary of this function goes here
%   Detailed explanation goes here

    for i=1:length(avg_channels_location)
        if(i > length(avg_channels_location))
            return
        end
        current_label = avg_channels_location(i);
        is_found = 0;
        for j=1:length(channels_location)
           if(strcmp(channels_location(j).labels,current_label))
               is_found = 1;
               break;
           end
        end
        
        if(is_found == 0)
            avg_channels_location(i) = [];
            i = i - 1;
        end
    end
end

function  [channels_location] = filter_non_scalp(channels_location)
%FILTER_NON_SCALP Summary of this function goes here
%   Detailed explanation goes here
    non_scalp_channel_label = {'E127', 'E126', 'E17', 'E128', 'E125', 'E21', 'E25', 'E32', 'E38', 'E44', 'E14', 'E8', 'E1', 'E121', 'E114', 'E43', 'E49', 'E56', 'E63', 'E68', 'E73', 'E81', 'E120', 'E113', 'E107', 'E99', 'E94', 'E88', 'E48', 'E119'};

    for i=1:length(non_scalp_channel_label)
        current_label = non_scalp_channel_label{i};
        for j=1:length(channels_location)
           if(strcmp(channels_location(j).labels,current_label))
               channels_location(j) = [];
               break;
           end
        end
    end
end



function  [motifs] = fit_motifs(motifs,channels_location, avg_channels_location)
%FILTER_NON_SCALP Summary of this function goes here
%   Detailed explanation goes here

    for i=1:length(channels_location)
        if(i > length(channels_location))
            return
        end
        current_label = channels_location(i);
        is_found = 0;
        for j=1:length(avg_channels_location)
           if(strcmp(avg_channels_location(j).labels,current_label))
               is_found = 1;
               break;
           end
        end
        
        if(is_found == 0)
            channels_location(i) = [];
            motifs.frequency(:,i) = [];
            motifs.source(:,i) = [];
            motifs.distance(:,i) = [];
            i = i - 1;
        end
    end
end

function [avg_motifs, motifs_count] = add_motifs(avg_motifs, avg_channels_location,e_i, motifs, channels_location, motifs_count)

    for i=1:length(avg_channels_location)
        current_label = avg_channels_location(i).labels;
        is_found = 0;
        for j=1:length(channels_location)
           if(strcmp(channels_location(j).labels, current_label))
               is_found = j;
               break;
           end
        end
        
        if(is_found ~= 0)
            j = is_found;
            for m_i = 1:13
                % if the sum of the channels frequency is bigger than 0 it
                % means we have a significant motif
                if(sum(motifs.frequency(m_i,:)) > 0)
                    avg_motifs.frequency(e_i,m_i,i) = avg_motifs.frequency(e_i,m_i,i) + motifs.frequency(m_i,j);
                    avg_motifs.source(e_i,m_i,i) = avg_motifs.source(e_i,m_i,i) + motifs.source(m_i,j);
                    avg_motifs.distance(e_i,m_i,i) = avg_motifs.distance(e_i,m_i,i) + motifs.distance(m_i,j);
                    motifs_count(e_i,m_i,i) = motifs_count(e_i,m_i,i) + 1;
                end
            end
        end
    end
end
