%{
    Yacine Mahdid 2020-01-08
    This script will calculate the motif on the dpli matrices at
    alpha generated previously in the experiment 14. It will save them at
    the same place as the pli matrices. This script use the augmented
    version of the motif analysis which will calculate the distance and the
    sink information.

    * Warning: This experiment use the setup_experiments.m script to 
    load variables. Therefore if you are trying to edit this code and you
    don't know what a variable mean take a look at the setup_experiments.m
    script.
%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

% Create the motif output directory
motif_output_path = mkdir_if_not_exist(output_path,'motif');
dpli_input_path = strcat(ouput_path,filesep,'dpli');
% Iterate over the participants
for p = 1:length(participants)

    % Create the participants directory
    participant = participants{p};
    motif_participant_output_path =  mkdir_if_not_exist(motif_output_path,participant);
    dpli_participant_input_path = strcat(dpli_input_path,filesep,participant);
    
    % Iterate over the states
    for s = 1:length(states)
        state = states{s};
        
        motif_state_filename = strcat(motif_participant_ouput_path,filesep,state,'_motif.mat');
        
        % Load the wpli result
        data = load(strcat(dpli_participant_input_path,filesep,state,'.mat')); 
        result_dpli = data.result_dpli;
        dpli_matrix  = result_dpli.data.avg_dpli;
        channels_location = result_dpli.metadata.channels_location;
        
        % Transform the dpli into phase lead
        phase_lead_matrix = make_phase_lead(dpli_matrix);
        
        % Filter the channels location to match the filtered motifs
        [phase_lead_matrix,channels_location] = filter_non_scalp(phase_lead_matrix,channels_location);
        
        % Calculate motif with 3 connection
        [frequency, source, target, distance] = motif_3(phase_lead_matrix, ... 
                                                channels_location, motif_param.number_rand_network, ...
                                                motif_param.bin_swaps, motif_param.weight_frequency);
        
        % Save the motif data into a structure and into disk
        % we need all the information for the next experiments
        result_motif = struct();
        result_motif.channels_location = channels_location;
        result_motif.dpli_matrix = dpli_matrix;
        result_motif.phase_lead_matrix = phase_lead_matrix;
        result_motif.frequency = frequency;
        result_motif.source = source;
        result_motif.target = target;
        result_motif.distance = distance;
        
        save(motif_state_filename, 'result_motif');
        
    end
end