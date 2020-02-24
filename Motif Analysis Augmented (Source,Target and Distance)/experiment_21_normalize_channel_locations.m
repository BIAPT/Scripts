%{
 Danielle Nadin 2020-02-24
 Loop through MDFA data and change channel location labels of dpli, wpli
 and power results (motif, hubs and graph theory can all be fixed by
 rerunning the code to generate motif, hubs and graph theory)
%}

%% Seting up the variables
clear;
setup_experiments % see this file to edit the experiments

dpli_input_path = strcat(output_path,filesep,'dpli');
wpli_input_path = strcat(output_path,filesep,'wpli');
power_input_path = strcat(output_path,filesep,'power');

% Iterate over the participants
for p = 1:length(participants)
    
    % Create the participants directory
    participant = participants{p};
    disp(strcat("Participant : ",participant));

    dpli_participant_input_path = strcat(dpli_input_path,filesep,participant);
    wpli_participant_input_path = strcat(wpli_input_path,filesep,participant); 
    power_participant_input_path = strcat(power_input_path,filesep,participant); 
    
    % Iterate over the states
    for s = 1:length(states)
        state = states{s};
        dpli_state_filename = strcat(dpli_participant_input_path,filesep,state,'_dpli.mat');
        wpli_state_filename = strcat(wpli_participant_input_path,filesep,state,'_wpli.mat');
        power_state_filename = strcat(power_participant_input_path,filesep,state,'_power.mat');
        
        % Load the dpli and wpli result
        load(dpli_state_filename);
        load(wpli_state_filename);
        load(power_state_filename);
        ppt_location = result_dpli.metadata.channels_location;

        %Rename channels 
        for i = 1:length(ppt_location)
            t_label = ppt_location(i).labels;
            if strcmp(t_label,'Fp2')
                ppt_location(i).labels = 'E9';
            elseif strcmp(t_label,'Fz')
                ppt_location(i).labels = 'E11';
            elseif strcmp(t_label,'Fp1')
                ppt_location(i).labels = 'E22';
            elseif strcmp(t_label,'F3')
                ppt_location(i).labels = 'E24';
            elseif strcmp(t_label,'F7')
                ppt_location(i).labels = 'E33';
            elseif strcmp(t_label,'C3')
                ppt_location(i).labels = 'E36';
            elseif strcmp(t_label,'T7')
                ppt_location(i).labels = 'E45';
            elseif strcmp(t_label,'P3')
                ppt_location(i).labels = 'E52';
            elseif strcmp(t_label,'LM')
                ppt_location(i).labels = 'E57';
            elseif strcmp(t_label,'P7')
                ppt_location(i).labels = 'E58';
            elseif strcmp(t_label,'Pz')
                ppt_location(i).labels = 'E62';
            elseif strcmp(t_label,'O1')
                ppt_location(i).labels = 'E70';
            elseif strcmp(t_label,'Oz')
                ppt_location(i).labels = 'E75';
            elseif strcmp(t_label,'O2')
                ppt_location(i).labels = 'E83';
            elseif strcmp(t_label,'P4')
                ppt_location(i).labels = 'E92';
            elseif strcmp(t_label,'P8')
                ppt_location(i).labels = 'E96';
            elseif strcmp(t_label,'RM')
                ppt_location(i).labels = 'E100';
            elseif strcmp(t_label,'C4')
                ppt_location(i).labels = 'E104';
            elseif strcmp(t_label,'T8')
                ppt_location(i).labels = 'E108';
            elseif strcmp(t_label,'F8')
                ppt_location(i).labels = 'E122';
            elseif strcmp(t_label,'F4')
                ppt_location(i).labels = 'E124';
            end
        end
        
        %Replace channel location file in dpli and wpli result structure
        result_dpli.metadata.channels_location = ppt_location;
        result_wpli.metadata.channels_location = ppt_location;
        result_td.metadata.channels_location = ppt_location;
        save(dpli_state_filename, 'result_dpli');
        save(wpli_state_filename, 'result_wpli');
        save(power_state_filename, 'result_td');
    end
end


