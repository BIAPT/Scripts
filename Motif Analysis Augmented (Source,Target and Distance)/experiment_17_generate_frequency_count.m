%{
    Yacine Mahdid 2020-01-22
    %Modified by Danielle Nadin on 2020-02-03
    This script is used to generate the motif frequencies we need
    for the paper. 

    The output should be a csv file with the same format as we need to run
    the statistics on

%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

data_output_file = strcat(output_path,filesep,'motif',filesep,'motif_frequency.csv');
motif_data_path = strcat(output_path,filesep,'motif');

% Header for output file
header = {'subject_id','state', 'motif_id', 'frequency'};
motif_ids = [1,7];

%% Calculating and Writing the frequency to CSV

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
    
    for m = 1:length(motif_ids)
        motif_id = motif_ids(m);
        
        for s = 1:length(states)
            state = states{s};
            
            % Load the data
            data_filename = strcat(state,'_motif.mat');
            data_location = strcat(motif_data_path, filesep, participant, filesep, data_filename);
            data = load(data_location);
            motif = data.result_motif;
            
            % Get frequency
            freq = sum(motif.frequency(motif_id, :));
           
            % Print to the file 
            fprintf(file_id,"%s,%s,%s,%f,%f,%f\n",participant, state,...
                string(motif_id),freq);
            fprintf(file_id,"\n");
        end
    end
end

% Close the file pointer when done 
fclose(file_id);
