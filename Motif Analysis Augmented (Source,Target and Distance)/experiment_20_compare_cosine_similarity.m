%{
    Yacine Mahdid 2020-01-22
    This script is used to generate the average motif across participants

    Modified by Danielle Nadin 2020-02-24
    Add ability to compute cosine similarity only on anterior or posterior
    electrodes. 

%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

%% Setup the variables
num_motif = 13;
motif_folder = 'D:\DOC\Motif analysis\Results\Alpha\motif';

%% Manual Variable to Change
ppt_name_1 = 'WSAS13';
ppt_state_1 = '_Pre_5min';

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
    
    % Get the anterior data points
    vector_1_anterior = get_anterior(vector_1, ppt_data_1.channels_location); %use the patient channel locations (< 99 chans)
    vector_2_anterior = get_anterior(vector_2, ppt_data_1.channels_location);
    
    % Get the posterior data points
    vector_1_posterior = get_posterior(vector_1, ppt_data_1.channels_location);
    vector_2_posterior = get_posterior(vector_2, ppt_data_1.channels_location);
    
    if (sum(vector_1) ~= 0 && sum(vector_2) ~=0)
        cosine_similarity = vector_cosine_similarity(vector_1,vector_2); 
        cos_anterior = vector_cosine_similarity(vector_1_anterior, vector_2_anterior);
        cos_posterior = vector_cosine_similarity(vector_1_posterior, vector_2_posterior);
        disp(strcat("Motif: ", string(m)," Cosine similarity (whole brain) = ", string(cosine_similarity)))
        disp(strcat("Motif: ", string(m)," Cosine similarity (anterior) = ", string(cos_anterior)))
        disp(strcat("Motif: ", string(m)," Cosine similarity (posterior) = ", string(cos_posterior)))
        disp(strcat("Motif: ", string(m)," Cosine similarity (anterior-posterior average) = ", string(mean([cos_anterior cos_posterior]))))
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
