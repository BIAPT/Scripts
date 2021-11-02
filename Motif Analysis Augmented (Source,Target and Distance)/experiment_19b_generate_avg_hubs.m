%{
    Yacine Mahdid 2020-01-22
    This script is used to generate the average motif across participants

    Modified by Danielle Nadin 2020-02-19
    Modified to generate average hubs instead of motifs

%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

%% Setup the variables
hubs_folder = 'D:\Motif analysis\MDFA\Results\Alpha\hubs';
[avg_hubs_path] = mkdir_if_not_exist(hubs_folder, 'AVERAGE');
participants = {'MDFA03','MDFA05','MDFA06','MDFA07','MDFA10','MDFA11','MDFA12','MDFA15','MDFA17'};
states = {'BASELINE','IF5','EMF5','EML30','EML10','EML5','EC3','RECOVERY'};

%% Iterate over the state
for s = 1:length(states)
    state = states{s};
    
    % Creating empty total participant structure
    total_ppt_hubs = struct();
    total_ppt_hubs.channels_location = zeros(1,1);
    total_ppt_hubs.degree = zeros(1,1);
    total_ppt_hubs.num_participant = zeros(1,1);
    
    for p = 1:length(participants)
        participant = participants{p};
        
        hubs_path = strcat(hubs_folder,filesep,participant,filesep,state,'_hubs.mat');
        data = load(hubs_path);
        ppt_hubs = data.result_hubs;
        
        % Check if need to add to the total or make this the total
        if length(total_ppt_hubs.degree) == length(zeros(1,1))
            total_ppt_hubs.channels_location = ppt_hubs.channels_location;
            total_ppt_hubs.degree = ppt_hubs.degree;
            
            num_channel_ppt = length(ppt_hubs.channels_location);
            total_ppt_hubs.num_participant = zeros(1,num_channel_ppt);
            
        else
            total_ppt_hubs = add_hubs(total_ppt_hubs, ppt_hubs);
        end
    end
    % Averaging the hubs
    result_hubs = average_hubs(total_ppt_hubs);
    
    % Saving the hubs
    hubs_state_filename = strcat(avg_hubs_path,filesep,state,'_hubs.mat');
    save(hubs_state_filename, 'result_hubs');
   
    
end

    %% Helper functions
    function total_ppt_hubs = add_hubs(total_ppt_hubs, ppt_hubs)
    % TOTAL_PPT_HUBS will sum a participant hubs to the grand total
    % This is using channels location to know where to sum the data
    
    ppt_location = ppt_hubs.channels_location; % we make a copy so we can delete from here
    
    % Iterating over the total label to find if the participant has a
    % similar channel
    for tl_i = 1:length(total_ppt_hubs.channels_location)
        t_label = total_ppt_hubs.channels_location(tl_i).labels;
        
        for pl_i = 1:length(ppt_location)
            p_label = ppt_location(pl_i).labels;
            
            % They are the same so we add at the location tl_i in the
            % structure
            if(strcmp(t_label, p_label))
                
                
                total_ppt_hubs.degree(tl_i) = total_ppt_hubs.degree(tl_i) + ppt_hubs.degree(pl_i);
                total_ppt_hubs.num_participant(tl_i) = total_ppt_hubs.num_participant(tl_i) + 1;
                
                
                
                % Delete that location
                ppt_location(pl_i) = [];
                break;
            end
        end
    end
    
    % Iterate over the remainder of the ppt_location and add them at the
    % end of the grand total
    % Won't do
    
    end
    
    function avg_ppt_hubs = average_hubs(total_ppt_hubs)
    % AVG_PPT_HUBS will average the grand total by the number of participant
    
    avg_ppt_hubs = total_ppt_hubs;
    avg_ppt_hubs.degree = total_ppt_hubs.degree ./ total_ppt_hubs.num_participant;
    
    end
