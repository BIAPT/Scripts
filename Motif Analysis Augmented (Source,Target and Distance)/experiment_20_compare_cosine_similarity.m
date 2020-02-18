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
ppt_name_1 = 'MDFA03';
ppt_state_1 = 'EML10';

ppt_name_2 = 'AVERAGE';
ppt_state_2 = 'BASELINE';

%% loading the data
data = load(strcat(motif_folder,filesep,ppt_name_1,filesep,ppt_state_1,'_motif.mat'));
ppt_data_1 = data.result_motif;

data = load(strcat(motif_folder,filesep,ppt_name_2,filesep,ppt_state_2,'_motif.mat'));
ppt_data_2 = data.result_motif;

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

