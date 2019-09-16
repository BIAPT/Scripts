%Yacine Mahdid 2019-06-26 
% modified by Danielle Nadin 2019-09-13
% modified by Yacine Mahdid 2019-09-15

% Analysis done in the context of Network Hub Analysis to compare node degree from DOC patient
% with average healthy participants and across different states using cosine similarity. 
%% Static Variables
% Selector Labels
regions_labels = {'Anterior','Posterior'};

%% Load the data Using File Explorer

% Individual Participant data 1
title = "Select the single participant hub structure:";
print(title)
data = load_from_file(title);
hub_pre = data.result_pre;
hub_pre = (hub_pre - mean(hub_pre)) / (std(hub_pre));

% Individual Participant data 2
title = "Select another single participant hub structure:";
print(title)
data = load_from_file(title);
hub_other = data.result_post;
hub_other = (hub_other - mean(hub_other)) / (std(hub_other));


title = "Select the single participant EEG_info structure:";
print(title)
data = load_from_file(title);
individual_channels_location = data.EEG_info.chanlocs;


%% Pre-Processing before Cosine Similarity 

% Select the anterior or the posterior part of the brain for analysis
[hub_pre, hub_other] = select_region(hub_pre,hub_other,individual_channels_location,regions_labels);

%% Cosine Similarity
cosine_similarity = vector_cosine_similarity(hub_pre,hub_other);

print("Cosine similarity: ");
disp(cosine_similarity);


