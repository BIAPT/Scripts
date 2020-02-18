%{
    Yacine Mahdid 2020-01-22
    This script is used to generate the average motif across participants

%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

%% Setup the variables
num_motif = 13;
motif_folder = '/home/yacine/Documents/motif_analysis_output/motif';

%% Manual Variable to Change
ppt_name_1 = 'MDFA07';
ppt_state_1 = 'EML10';

ppt_name_2 = 'AVERAGE';
ppt_state_2 = 'BASELINE';

%% loading the data
data = load(strcat(motif_folder,filesep,ppt_name_1,filesep,ppt_state_1,'_motif.mat'));
ppt_data_1 = data.result_motif;

data = load(strcat(motif_folder,filesep,ppt_name_2,filesep,ppt_state_2,'_motif.mat'));
ppt_data_2 = data.result_motif;

%% Equalizing the number of data points
[ppt_data_1, ppt_data_2] = equalize(ppt_data_1, ppt_data_2);

message = strcat("Participant: ",ppt_name_1," at ", ppt_state_1, " against ", ppt_name_2, " at ", ppt_state_2);
disp(message)
for m = 1:num_motif
    vector_1 = normalize(ppt_data_1.frequency(m,:));
    vector_2 = normalize(ppt_data_2.frequency(m,:));
    
    if (sum(vector_1) ~= 0 && sum(vector_2) ~=0)
        cosine_similarity = vector_cosine_similarity(vector_1,vector_2); 
        disp(strcat("Motif: ", string(m)," Cosine similarity = ", string(cosine_similarity)))
    end
end


function [ppt_data_1, ppt_data_2] = equalize(ppt_data_1, ppt_data_2)
    

    % If both data structure have the same amount of channels
    if (length(ppt_data_1.channels_location) == length(ppt_data_2.channels_location))
       return 
    end
    

    index_to_remove = [];
    for p1_i = 1:length(ppt_data_1.channels_location)
       label_1 = ppt_data_1.channels_location(p1_i).labels;

       is_found = 0;
       for p2_i = 1:length(ppt_data_2.channels_location)
          label_2 = ppt_data_2.channels_location(p2_i).labels;

          if(strcmp(label_1,label_2))
             is_found = 1;
             break;
          end
       end

       if(is_found == 0)
          index_to_remove = [index_to_remove, p1_i];
       end
    end
    
    ppt_data_1.frequency(:,index_to_remove) = [];
    
    index_to_remove = [];
    for p2_i = 1:length(ppt_data_2.channels_location)
       label_2 = ppt_data_2.channels_location(p2_i).labels;

       is_found = 0;
       for p1_i = 1:length(ppt_data_1.channels_location)
          label_1 = ppt_data_1.channels_location(p1_i).labels;

          if(strcmp(label_1, label_2))
             is_found = 1;
             break;
          end
       end

       if(is_found == 0)
          index_to_remove = [index_to_remove, p2_i];
       end
    end
    
    ppt_data_2.frequency(:,index_to_remove) = [];
end