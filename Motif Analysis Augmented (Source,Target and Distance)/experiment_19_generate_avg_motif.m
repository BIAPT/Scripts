%{
    Yacine Mahdid 2020-01-22
    This script is used to generate the average motif across participants

%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

%% Setup the variables
num_motif = 13;
motif_folder = 'D:\Motif analysis\MDFA\Results\Alpha\motif';
[avg_motif_path] = mkdir_if_not_exist(motif_folder, 'AVERAGE');
participants = {'MDFA03','MDFA05','MDFA06','MDFA07','MDFA10','MDFA11','MDFA12','MDFA15','MDFA17'};
states = {'BASELINE','EMF5','EC3','RECOVERY'};

%% Iterate over the state
for s = 1:length(states)
   state = states{s};
   
   % Creating empty total participant structure
   total_ppt_motif = struct();
   total_ppt_motif.channels_location = zeros(1,1);
   total_ppt_motif.frequency = zeros(1,1);
   total_ppt_motif.source = zeros(1,1);
   total_ppt_motif.target = zeros(1,1);
   total_ppt_motif.distance = zeros(1,1);
   total_ppt_motif.num_participant = zeros(1,1);
   
   for p = 1:length(participants)
      participant = participants{p};
      
      motif_path = strcat(motif_folder,filesep,participant,filesep,state,'_motif.mat');
      data = load(motif_path);
      ppt_motif = data.result_motif;
      
      % Check if need to add to the total or make this the total
      if length(total_ppt_motif.frequency) == length(zeros(1,1))
          total_ppt_motif.channels_location = ppt_motif.channels_location;
          total_ppt_motif.frequency = ppt_motif.frequency;
          total_ppt_motif.source = ppt_motif.source;
          total_ppt_motif.target = ppt_motif.target;
          total_ppt_motif.distance = ppt_motif.distance;
          
          num_channel_ppt = length(ppt_motif.channels_location);
          total_ppt_motif.num_participant = zeros(num_motif,num_channel_ppt);  
          
          % Correction for the num_participant contribution when not
          % significant
          for m = 1:num_motif
              % If the motif wasn't significant we count as if the
              % participant isn't contributing to it.
              if sum(ppt_motif.frequency(m,:)) == 0
                total_ppt_motif.num_participant(m,:) = zeros(1,num_channel_ppt);
              end
          end

      else
          total_ppt_motif = add_motif(total_ppt_motif, ppt_motif, num_motif);
      end
      
   end
   
   % Averaging the motifs
   result_motif = average_motif(total_ppt_motif, num_motif);
   
   % Saving the motif
   motif_state_filename = strcat(avg_motif_path,filesep,state,'_motif.mat');
   save(motif_state_filename, 'result_motif');
end

%% Helper functions
function total_ppt_motif = add_motif(total_ppt_motif, ppt_motif, num_motif)
% TOTAL_PPT_MOTIF will sum a participant motif to the grand total
% This is using channels location to know where to sum the data

    ppt_location = ppt_motif.channels_location; % we make a copy so we can delete from here

    % Iterating over the total label to find if the participant has a
    % similar channel
    for tl_i = 1:length(total_ppt_motif.channels_location)
        t_label = total_ppt_motif.channels_location(tl_i).labels;
        
        for pl_i = 1:length(ppt_location)
            p_label = ppt_location(pl_i).labels;
            
            % They are the same so we add at the location tl_i in the
            % structure
            if(strcmp(t_label, p_label))
                
                for m = 1:num_motif
                    % If the motif is significant at this index this is
                    % where we will add everything to the total
                    if (sum(ppt_motif.frequency(m,:)) ~= 0)
                        total_ppt_motif.frequency(m,tl_i) = total_ppt_motif.frequency(m, tl_i) + ppt_motif.frequency(m,pl_i);
                        total_ppt_motif.source(m,tl_i) = total_ppt_motif.source(m, tl_i) + ppt_motif.source(m,pl_i);
                        total_ppt_motif.target(m,tl_i) = total_ppt_motif.target(m, tl_i) + ppt_motif.target(m,pl_i);
                        total_ppt_motif.distance(m,tl_i) = total_ppt_motif.distance(m, tl_i) + ppt_motif.distance(m,pl_i);
                        total_ppt_motif.num_participant(m, tl_i) = total_ppt_motif.num_participant(m, tl_i) + 1;
                    end 
                end

                
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

function avg_ppt_motif = average_motif(total_ppt_motif, num_motif)
% AVG_PPT_MOTIF will average the grand total by the number of participant
   
    avg_ppt_motif = total_ppt_motif;
    % If the motif is significant at this index this is
    % where we will add everything to the total
    for m = 1:num_motif
        if (sum(total_ppt_motif.frequency(m,:)) ~= 0)
            avg_ppt_motif.frequency(m,:) = total_ppt_motif.frequency(m,:) ./ total_ppt_motif.num_participant(m,:);
            avg_ppt_motif.source(m,:) = total_ppt_motif.source(m,:) ./ total_ppt_motif.num_participant(m,:);
            avg_ppt_motif.target(m,:) = total_ppt_motif.target(m,:) ./ total_ppt_motif.num_participant(m,:);
            avg_ppt_motif.distance(m,:) = total_ppt_motif.distance(m,:) ./ total_ppt_motif.num_participant(m,:);
        end
        
    end
end