%Yacine Mahdid 2019-08-13

% Analysis done in the context of my Technical Paper for the first review
% In this script I will be calculating cosine similarity on the topographic
% map for the downsampled eeg data as compared to the other headset eeg
% data


%% Load the data Using File Explorer

title = "Select the data for the downsampled egi headset:";
print(title)
data = load_from_file(title);
topographic_map_egi = normalize(data.topodata);

title = "Select another data for the single trial headset:";
print(title)
data = load_from_file(title);
topographic_map_headset = normalize(data.topodata);

cosine_similarity = vector_cosine_similarity(topographic_map_egi,topographic_map_headset);

print(strcat("Cosine similarity = ", string(cosine_similarity)));

function normalized_vector = normalize(vector)
    normalized_vector = (vector - min(vector))/(max(vector)-min(vector));
 end