%Yacine Mahdid 2019-07-04

% Analysis done in the context of Motif Analysis to compare motif frequency from DOC patient
% across different condition 

%% Static Variables
% Selector Labels
motif_labels = {'M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13'};
regions_labels = {'Anterior','Posterior'};

%% Load the data Using File Explorer

% Individual Participant data 1
title = "Select the single participant motif analysis structure:";
print(title)
data = load_from_file(title);
individual_motifs_1 = data.motifs.node_frequency;

% Individual Participant data 2
title = "Select another single participant motif analysis structure:";
print(title)
data = load_from_file(title);
individual_motifs_2 = data.motifs.node_frequency;


title = "Select the single participant EEG_info structure:";
print(title)
data = load_from_file(title);
individual_channels_location = data.EEG_info.chanlocs;
%% User selection of the Average Participant Data

% Select the motif to compare between participant
print("Select the motif:");
[individual_motif_1,individual_motif_2] = select_motif(individual_motifs_1,individual_motifs_2,motif_labels);

%% Pre-Processing before Cosine Similarity 

% Normalization of both the individual motif
individual_motif_1 = normalize(individual_motif_1);
individual_motif_2 = normalize(individual_motif_2);

% Select the anterior or the posterior part of the brain for analysis
[individual_motif_1, individual_motif_2] = select_region(individual_motif_1,individual_motif_2,individual_channels_location,regions_labels);

%% Cosine Similarity
cosine_similarity = vector_cosine_similarity(individual_motif_1,individual_motif_2);

print("Cosine similarity: ");
disp(cosine_similarity);


