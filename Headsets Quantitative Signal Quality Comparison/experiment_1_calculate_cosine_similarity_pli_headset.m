%Yacine Mahdid 2019-08-09

% Analysis done in the context of my Technical Paper for the first review


%% Load the data Using File Explorer

title = "Select the data for the downsampled egi:";
print(title)
data = load_from_file(title);
connectivity_egi = normalize(data.z_score);

title = "Select another data for the actual headset";
print(title)
data = load_from_file(title);
connectivity_headset = normalize(data.z_score);

cosine_similarity = vector_cosine_similarity(connectivity_egi,connectivity_headset);

print(strcat("Cosine similarity = ", string(cosine_similarity)));

function normalized_vector = normalize(matrix)
    %Create the vector without the midline
    vector = [];
    for i = 1:length(matrix)
        for j = 1:length(matrix)
            if(i ~= j)
               vector = [vector matrix(i,j)]; 
            end
        end 
    end
    
    normalized_vector = vector;
    
    %normalized_vector = (vector - min(vector))/(max(vector)-min(vector));
 end