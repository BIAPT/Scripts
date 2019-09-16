%Yacine Mahdid 2019-06-26 
% modified by Danielle Nadin 2019-09-13
% modified by Yacine Mahdid 2019-09-15
% modified by Danielle Nadin 2019-09-16

% Analysis done in the context of Network Hub Analysis to compare node degree from DOC patient
% with average healthy participants and across different states using cosine similarity. 
%% Static Variables and frequency selection
% Selector Labels
regions_labels = {'Anterior','Posterior'};
frequency_labels = {'Alpha','Theta','Delta'};

%Paths
average_info_path = strcat('data',filesep,'EEG_info_AVG.mat');

%% Load the data Using File Explorer

% Select the Frequency to compare
print("Select the frequency:");
average_degrees = select_frequency(frequency_labels);
average_degrees = (average_degrees - mean(average_degrees)) / (std(average_degrees));

% Average Participant channel locations
data = load(average_info_path);
average_channels_location = data.EEG_info.chanlocs;

% Individual Participant data 
title = "Select the single participant hub structure:";
print(title)
data = load_from_file(title);
hub_pre = data.degrees;
hub_pre = (hub_pre - mean(hub_pre)) / (std(hub_pre));

title = "Select the single participant EEG_info structure:";
print(title)
data = load_from_file(title);
individual_channels_location = data.EEG_info.chanlocs;

%% Pre-Processing before Cosine Similarity 

% Equalize the average data to the individual participant data
[average_degrees,average_channels_location] = remove_channels(average_degrees,average_channels_location,individual_channels_location);

% Equalize the individual data to the left over average participant data
[hub_pre, individual_channels_location] = remove_channels(hub_pre,individual_channels_location,average_channels_location);

% Select the anterior or the posterior part of the brain for analysis
[average_degrees, hub_pre] = select_region(average_degrees,hub_pre,individual_channels_location,regions_labels);

%% Cosine Similarity
cosine_similarity = vector_cosine_similarity(average_degrees,hub_pre);

print("Cosine similarity: ");
disp(cosine_similarity);


