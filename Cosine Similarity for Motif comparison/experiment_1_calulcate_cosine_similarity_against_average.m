%Yacine Mahdid 2019-06-26

% Analysis done in the context of Motif Analysis to compare motif frequency from DOC patient
% with average healthy participants using cosine similarity. 

%% Static Variables
% Selector Labels
epoch_labels = {'EC1','IF5','EF5','EL30','EL10','EL5','EC3','EC4','EC5','EC6','EC7','EC8'};
frequency_labels = {'Alpha','Theta'};
motif_labels = {'M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13'};
regions_labels = {'Anterior','Posterior'};

%Paths
average_data_path = strcat('data',filesep,'average_data.mat');
average_info_path = strcat('data',filesep,'EEG_info_AVG.mat');

%% Load the data Using File Explorer

% Average Participant
data = load(average_data_path);
average_motifs = data.data_motifs;

data = load(average_info_path);
average_channels_location = data.EEG_info.chanlocs;

% Individual Participant
title = "Select the single participant motif analysis structure:";
print(title)
data = load_from_file(title);
individual_motifs = data.motifs.node_frequency;

title = "Select the single participant EEG_info structure:";
print(title)
data = load_from_file(title);
individual_channels_location = data.EEG_info.chanlocs;

%% User selection of the Average Participant Data

% Select the Epoch for average particiant
print("Select the epoch:")
average_motifs = select_epoch(average_motifs,epoch_labels);

% Select the Frequency to compare
print("Select the frequency:");
average_motifs = select_frequency(average_motifs,frequency_labels);

% Select the motif to compare between participant
print("Select the motif:");
[average_motif,individual_motif] = select_motif(average_motifs,individual_motifs,motif_labels);

%% Pre-Processing before Cosine Similarity 

% Normalization of the individual motif
individual_motif = normalize(individual_motif);

% Equalize the average data to the individual participant data
[average_motif,average_channels_location] = remove_channels(average_motif,average_channels_location,individual_channels_location);

% Equalize the individual data to the left over average participant data
[individual_motif, individual_channels_location] = remove_channels(individual_motif,individual_channels_location,average_channels_location);

% Select the anterior or the posterior part of the brain for analysis
[average_motif, individual_motif] = select_region(average_motif,individual_motif,individual_channels_location,regions_labels);

%% Cosine Similarity
cosine_similarity = vector_cosine_similarity(average_motif,individual_motif);
print("Cosine similarity: ");
disp(cosine_similarity);


