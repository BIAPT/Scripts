%Yacine Mahdid 2019-08-09

% Analysis done in the context of my Technical Paper for the first review


%% Load the data Using File Explorer

title = "Select the data for the Eyes open downsampled egi:";
print(title)
data = load_from_file(title);
connectivity_egi_open = normalize(data.z_score);

title = "Select the data for the Eyes closed downsampled egi:";
print(title)
data = load_from_file(title);
connectivity_egi_closed = normalize(data.z_score);

difference_egi = connectivity_egi_open - connectivity_egi_closed;

title = "Select eyes open for the actual headset";
print(title)
data = load_from_file(title);
connectivity_headset_open = normalize(data.z_score);

title = "Select eyes closed for the actual headset";
print(title)
data = load_from_file(title);
connectivity_headset_closed = normalize(data.z_score);

difference_headset = connectivity_headset_open - connectivity_headset_closed;

cosine_similarity = vector_cosine_similarity(difference_egi,difference_headset);

print(strcat("Cosine similarity = ", string(cosine_similarity)));


function normalized_vector = normalize(matrix)
    vector_length = length(matrix)*length(matrix);
    vector = reshape(matrix,[1 vector_length]);
    normalized_vector = vector; %normalized_vector = (vector - min(vector))/(max(vector)-min(vector));
 end