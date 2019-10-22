%{
    Goal here is to use the pre_process_participant function to process
    the data and build a .CSV file containing all the features for the ml
    part of the proces

    The data is formated in s1 to s32
%}

%% Variable setup
data_filename = "data.csv";
number_participants = 3;


data_location = "C:\Users\biapt\Documents\GitHub\Scripts\IEEE Brain Challenge";
header = ["participant_id", "trial_number", "window_number","valence", "arousal", "dominance", "liking"];

for channel_i = 1:32
        feature_id = strcat("channel ", string(channel_i));
        header = [header,feature_id];      
end


% Overwrite the file
delete(data_filename);

% Write header to the features file
fileID = fopen(data_filename,'w');
for i = 1:(length(header)-1)
    fprintf(fileID,'%s,',header(i));
end
fprintf(fileID,"%s\n",header(length(header)));
fclose(fileID);

% Append each features
names = generate_participant_names(number_participants);
for p_index = 1:number_participants
    disp(strcat("Participants #", string(p_index)));
    [participant_features, participant_labels] = pre_process_participant(names(p_index),data_location);
    [num_trials, num_windows, num_features] = size(participant_features);
    
    % Here we append the features to the csv file
    for t_index = 1:num_trials
       current_labels = squeeze(participant_labels(t_index,:));
       for w_index = 1:num_windows            
           % Write the features and labels
           current_features = squeeze(participant_features(t_index,w_index,:))';
           dlmwrite(data_filename, [p_index, t_index, w_index, current_labels, current_features], '-append');
       end
    end
end

%% Helper functions
% function to generate all the names we need
function [participant_names] = generate_participant_names(number_participants)
    participant_names = [];
    for i = 1:number_participants
       participant_names = [participant_names, strcat("s",string(i)),]; 
    end
end
