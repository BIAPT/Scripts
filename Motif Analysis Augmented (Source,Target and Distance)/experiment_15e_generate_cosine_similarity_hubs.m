%{
    Modified by Danielle Nadin 2020-02-18 modify computing cosine
    similarity on hubs instead of motifs
    Yacine Mahdid 2020-01-22
    This script is used to generate the cosine similarities value we need
    for the paper. This mirror the analysis we did in Cosine Similarity for Motif comparison
    But we automate everything now
    
    The output should be a csv file with the same format as we need to run
    the statistics on

%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

data_output_file = strcat(output_path,filesep,'hubs',filesep,'cosine_similarity.csv');
hubs_data_path = strcat(output_path,filesep,'hubs');

% Header for output file
header = {'subject_id','state', 'cos_anterior', 'cos_posterior', 'cos_whole'};

%% Calculating and Writing the cosine similarity to CSV

% Overwrite the file (useful for debuging when the code crash in the middle
% otherwise the file is not easy to open)
delete(data_output_file);

% Open the file in write mode
file_id = fopen(data_output_file,'w');

% Create the header
for i = 1:length(header)
    fprintf(file_id,'%s,',header{i});
end
fprintf(file_id,"\n");

% Iterate over all the data participant->motifs->states
for p = 1:length(participants)
    participant = participants{p};
        
        for s = 1:length(states)
            state = states{s};
            
            % Load the baseline data
            base_filename = strcat('BASELINE_hubs.mat');
            base_location = strcat(hubs_data_path, filesep, participant, filesep, base_filename);
            data = load(base_location);
            base_hubs = data.result_hubs;
            
            % Load the other data points (which might be baseline too)
            other_filename = strcat(state,'_hubs.mat');
            other_location = strcat(hubs_data_path, filesep, participant, filesep, other_filename);
            data = load(other_location);
            other_hubs = data.result_hubs;
            
            % Double checking that we have two vector of similar size
            [base_row, base_col] = size(base_hubs.channels_location);
            [other_row, other_col] = size(other_hubs.channels_location);
            assert(base_row == other_row & base_col == other_col);
            
            % Here we are comparing the baseline against every other state
            % (even against itself)
            deg_base = normalize(base_hubs.degree);
            deg_other = normalize(other_hubs.degree);
            location = other_hubs.channels_location;
            
            % Get the anterior data points
            deg_base_anterior = get_anterior(deg_base, location);
            deg_other_anterior = get_anterior(deg_other, location);
            
            % Get the posterior data points
            deg_base_posterior = get_posterior(deg_base, location);
            deg_other_posterior = get_posterior(deg_other, location);
            
            % Actually calculating the cosine similarity
            cos_anterior = vector_cosine_similarity(deg_base_anterior, deg_other_anterior);
            cos_posterior = vector_cosine_similarity(deg_base_posterior, deg_other_posterior);
            cos_whole = vector_cosine_similarity(deg_base, deg_other);
            
            % Print to the file
            fprintf(file_id,"%s,%s,%f,%f,%f\n",participant, state,...
                cos_anterior, cos_posterior, cos_whole);
        end
 
end

% Close the file pointer when done 
fclose(file_id);


function [sifted_vector] = get_anterior(vector, channels_location)
%   GET_ANTERIOR getter function to fetch only the part of the vector that are
%   in the anterior part of the brain
%   vector: motif frequency count vector of length number of channels
%   channels_location: chanlocs data structure with channel information
%
%   sifted_vector: is the motif frequency count vector minus the channels
%   location who didn't meet the threshold.

    sifted_vector = []; 
    for i = 1:length(vector)
        
        % Every channels that are anterior to the center line of the
        % headset is defined as anterior
        if(channels_location(i).X > -0.001)
            sifted_vector = [sifted_vector, vector(i)];
        end
    end
end

function [sifted_vector] = get_posterior(vector, channels_location)
%   GET_POSTERIOR getter function to fetch only the part of the vector that are
%   in the posterior part of the brain
%   vector: motif frequency count vector of length number of channels
%   channels_location: chanlocs data structure with channel information
%
%   sifted_vector: is the motif frequency count vector minus the channels
%   location who didn't meet the threshold.

    sifted_vector = [];
    for i = 1:length(vector)
        
        % Every channels that are in the below the center line of the
        % headset is defined as posterior
        if(channels_location(i).X < 0.001)
            sifted_vector = [sifted_vector, vector(i)];
        end
    end
end
