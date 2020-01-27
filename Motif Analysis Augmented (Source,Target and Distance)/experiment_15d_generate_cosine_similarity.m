%{
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

data_output_file = strcat(output_path,filesep,'motif',filesep,'cosine_similarity.csv');
motif_data_path = strcat(output_path,filesep,'motif');

% Header for output file
header = {'subject_id','state', 'motif_id', 'cos_anterior', 'cos_posterior', 'cos_whole'};
motif_ids = [1,7];

%% Calculating and Writing the cosine similarity to CSV

% Overwrite the file
delete(data_output_file);

% Open the 
file_id = fopen(data_output_file,'w');

% Create the header
for i = 1:length(header)
    fprintf(file_id,'%s,',header{i});
end
fprintf(file_id,"\n");


for p = 1:length(participants)
    participant = participants{p};
    
    for m = 1:length(motif_ids)
        motif_id = motif_ids(m);
        
        for s = 1:length(states)
            state = states{s};
            
            % Here we are comparing the baseline against every other state
            % (even against itself)
            cos_anterior = 1;
            cos_posterior = 1;
            cos_whole = 1;
            
            fprintf(file_id,"%s,%s,%s,%f,%f,%f\n",participant, state,...
                string(motif_id),cos_anterior, cos_posterior, cos_whole);
        end
    end
end
fclose(file_id);
