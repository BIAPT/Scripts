%% The goal here is to draft a functionality to reorder the channels

% The idea is to have a vector of labels that maps to the ordering we want
% to see
label_order = {'Fp1','Fp2','F3','F4','C3','Cz','C4','T5','T6','P3','Pz','P4','O1'};

% We are working with the data structure which has reduced_locations

% Setting up path variables
result_path = "/home/yacine/Documents/pain_and_eeg/results/msk/";

type = 'MSK Participants';

data = load(strcat(result_path,'MEAVG.mat'));
data = data.result;


function [ordered_matrix] = reorder_matrix(matrix, label_order, channels_location)
    % First step is to make a reordering vector
    % We'll iterate over the channel_location and find where it maps to the
    % label_order
    ordered_matrix = zeros(1,len(label_order));
    for i = 1:length(channels_location)
        
        % Get the label
        target_label = channels_location(i).labels; 
        % Get where it should be now
        target_index = find_item_index(target_label,label_vector);
        % Save it into the vector
        ordered_matrix(i) = target_index;
    end
end

% Helper function to find an item (target) in the label vector
function [index] = find_item_index(target, label_vector)
    index = -1;
    
    for i = 1:length(label_vector)
       if(strcmp(target,label_vector{i})
          index = i;
          return;
       end
    end
end