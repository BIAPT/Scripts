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
motif_ids = [1, 7]; 

% Used for the average figure
avg_data = struct();
avg_data.freq = zeros(8,13,99);
avg_data.source = zeros(8,13,99);
avg_data.dist = zeros(8,13,99);
avg_data.motifs_count = zeros(8,13,99);
avg_data.power_count = zeros(8,99);
avg_data.location = -1;

avg_data.power = zeros(8,99);

%% Motif Figure
motif_data_path = strcat(output_path,filesep,'motif');
power_data_path = strcat(output_path,filesep,'power');

for m = 1:length(motif_ids)
    motif_id = motif_ids(m);
    disp(strcat("Analyzing motif: ",string(motif_id)));
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

            % Load the power data
            power_data_filename = strcat(state,'_power.mat');
            power_data_location = strcat(power_participant_data_path,filesep,power_data_filename);
            data = load(power_data_location);
            result_td = data.result_td;

            if(sum(freq(:)) > 0)
                powers = cat(2,powers,result_td.data.filt_power);
                motif_freqs = cat(2,motif_freqs, freq); 
            end
            
            % Add the motifs to create the average figure afterward
            freq = result_motif.frequency;
            source = result_motif.source;
            dist = result_motif.distance;
            location = result_motif.channels_location;
            
            if(isstruct(avg_data.location) == 0)
                avg_data.location = location;
            end
            
            for e_i=1:length(avg_data.location)
                current_label = avg_data.location(e_i).labels;
                is_found = 0;
                for j=1:length(location)
                   if(strcmp(location(j).labels, current_label))
                       is_found = j;
                       break;
                   end
                end

                if(is_found ~= 0)
                    j = is_found;
                    for m_i = 1:13
                        % if the sum of the channels frequency is bigger than 0 it
                        % means we have a significant motif
                        if(sum(freq(m_i,:)) > 0)
                            avg_data.freq(s,m_i,e_i) = avg_data.freq(s,m_i, e_i) + freq(m_i, j);
                            avg_data.source(s,m_i,e_i) = avg_data.source(s,m_i, e_i) + source(m_i, j);
                            avg_data.dist(s,m_i,e_i) = avg_data.dist(s,m_i, e_i) + dist(m_i, j);
                            
                             
                            avg_data.motifs_count(s,m_i,e_i) = avg_data.motifs_count(s,m_i, e_i) + 1;
                        end
                    end
                    
                    avg_data.power(s,e_i) = avg_data.power(s, e_i) +  result_td.data.filt_power(j);
                    avg_data.power_count(s,e_i) = avg_data.power_count(s,e_i) + 1;
                end
            end
        end
    end

    %% Correlation Motif against Power Figure

    % Calculate first the correlation coefficient
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
    
    %% Generate the average figure
    
    for s = 1:length(states)
        for m_i = 1:13
            if(avg_data.motifs_count(s, m_i) ~= 0)
                for c_i = 1:99
                    avg_data.freq(s, m_i, c_i) = avg_data.freq(s, m_i, c_i) / avg_data.motifs_count(s, m_i, c_i);
                    avg_data.source(s ,m_i, c_i) = avg_data.source(s, m_i, c_i) / avg_data.motifs_count(s, m_i, c_i);
                    avg_data.dist(s, m_i, c_i) = avg_data.dist(s, m_i, c_i) / avg_data.motifs_count(s, m_i, c_i);   
                end
            end
        end
        
        for c_i = 1:99
            avg_data.power(s,c_i) = avg_data.power(s,c_i) / avg_data.power_count(c_i); 
        end
    end
    
    %{
    for e_i = 1:length(states)
        
        n_freq = normalize_motif(squeeze(avg_data.freq(e_i,:,:)));
        n_source = normalize_motif(squeeze(avg_data.source(e_i,:,:)));
        n_distance = normalize_motif(squeeze(avg_data.dist(e_i,:,:)));
        
        state = states{e_i};
        title_name = strcat("Average Participant at ", state, " for Motif ", string(motif_id));
        output_figure_path = strcat(figure_output_path,filesep,"Average_", state,'_motif_', string(motif_id),'.fig');

        plot_motif(n_freq(motif_id,:), n_source(motif_id,:), n_distance(motif_id,:), location, title_name);
        savefig(output_figure_path)
        close(gcf)
    end
    
    figure
    for e_i = 1:length(states)
        subplot(3,3,e_i)
        title(strcat("Average power at ",states{e_i}))
        topographic_map(avg_data.power(e_i,:),avg_data.location);
    end
    %}
    
end

function topographic_map(data,location)
    topoplot(data,location,'maplimits','absmax', 'electrodes', 'off');
    min_color = min(data);
    max_color = max(data);
    caxis([min_color max_color])
    colormap('jet')
    colorbar;
end