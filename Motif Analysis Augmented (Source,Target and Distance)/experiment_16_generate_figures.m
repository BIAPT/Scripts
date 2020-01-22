%{
    Yacine Mahdid 2020-01-22
    This script is used to generate all of the figures that are needed for
    the paper review.

    Note: We should call out three different way of generating the figures
    for the three different experiments we are conducting
%}

%% Seting up the variables
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

figure_output_path = mkdir_if_not_exist(output_path,'figure');

% Used for the correlation analysis
powers = [];
motif_freqs = [];        
motif_ids = [7]; % This is the motifs this code will be run against

%% Motif Figure
motif_data_path = strcat(output_path,filesep,'motif');
power_data_path = strcat(output_path,filesep,'power');
% Iterate over the participants
for p = 1:length(participants)

    % Create the participants directory
    participant = participants{p};
    disp(strcat("Participant: ", participant));
    
    figure_participant_output_path =  mkdir_if_not_exist(figure_output_path, participant);
    motif_participant_data_path = strcat(motif_data_path,filesep,participant);
    power_participant_data_path = strcat(power_data_path,filesep,participant);
    % Iterate over the states
    for s = 1:length(states)
        state = states{s};
        disp(strcat("State: ", state));
        
        % Load the motif data
        motif_data_filename = strcat(state,'_motif.mat');
        motif_data_location = strcat(motif_participant_data_path,filesep,motif_data_filename);
        data = load(motif_data_location);
        result_motif = data.result_motif;
        
        % Create the motif figure
        for m = 1:length(motif_ids)
           motif_id = motif_ids(m);
            freq = result_motif.frequency(motif_id, :);
            source = result_motif.source(motif_id, :);
            dist = result_motif.distance(motif_id, :);
            location = result_motif.channels_location;
            
            % Get the Z-Score of each values to plot in the topographic map
            n_freq = normalize_motif(freq);
            n_source = normalize_motif(source);
            n_distance = normalize_motif(dist); 

            title_name = strcat(participant," at ", state, " for Motif ", string(motif_id));
            output_figure_path = strcat(figure_participant_output_path,filesep,state,'_motif_',string(motif_id),'.fig');

            plot_motif(n_freq, n_source, n_distance, location, title_name);
            savefig(output_figure_path)
            close(gcf)
        end
        
        % Load the power data
        power_data_filename = strcat(state,'_power.mat');
        power_data_location = strcat(power_participant_data_path,filesep,power_data_filename);
        data = load(power_data_location);
        result_td = data.result_td;
        
        powers = cat(2,powers,result_td.data.filt_power);
        motif_freqs = cat(2,motif_freqs, freq); 
    end
end

%% Correlation Motif against Power Figure

% Calculate first the correlation coefficient
for m = 1:length(motif_ids)
    motif_id = motif_ids(m); 
    motif_frequency = motif_freqs(motif_id);
    [R,~] = corrcoef(motif_freqs,powers);% This gives a matrix of correlation
    correlation = R(2); % We just take the correlation between frequency and power and not power with power
    figure;
    scatter(motif_freqs,powers);
    xlabel('Motif Frequency');
    ylabel('Power')
    title(strcat("Power versus Motif ",string(motif_id)," Frequency at Alpha (R = ",string(correlation),")"));
    output_figure_path = strcat(figure_output_path, filesep, "correlation_power_motif_",string(motif_id),".fig");
    savefig(output_figure_path)
    close(gcf)
end
