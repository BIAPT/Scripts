%{
    Yacine Mahdid 2020-01-08
    This script is used to generate power spectra for the MDFA dataset.
    This was requested by the reviewer to make sure that the motif are not
    epiphenomenon of the shift in alpha power.
%}
 
%% Seting up the variables 
clear % to keep only what is needed for this experiment
setup_experiments % see this file to edit the experiments

% Create power output directory
power_output_path = mkdir_if_not_exist(output_path,'power');

% Create average result struct
% sp - spectral power (uV^2/Hz), for computing power ratios
% td - topographic distribution (dB), for plotting head maps
avg_data = struct();
avg_data.power_td = zeros(length(states),99);
avg_data.power_count = zeros(length(states),99);
avg_data.avg_power_td = zeros(length(states),99);
avg_data.location = -1;

% Iterate over the participants
for p = 1:length(participants)

    % Create the participants directory
    participant = participants{p};
    disp(strcat("Participant: ", participant));
    power_participant_output_path =  mkdir_if_not_exist(power_output_path, participant);

    % Iterate over the states
    for s = 1:length(states)
        state = states{s};
        disp(strcat("State: ", state));
        
        % Load the recording
        raw_data_filename = strcat(participant,state,'.set');
        data_location = strcat(raw_data_path,filesep,participant);
        recording = load_set(raw_data_filename,data_location);
        power_state_filename = strcat(power_participant_output_path,filesep,state,'_power.mat');
        
        % Calculate power
         
        % topographic distribution - units: dB (10*log10(spect/windows))
        % in this case, window size = full length of recording (300 sec)
        window_size = floor(recording.length_recording / recording.sampling_rate); % in seconds
        step_size = window_size; 
        result_td = na_topographic_distribution(recording, window_size, step_size, power_param.frequency_band);
        [filt_power, filt_location] = filter_non_scalp_vector(result_td.data.power, result_td.metadata.channels_location);
        result_td.data.filt_power = filt_power;
        result_td.data.filt_location = filt_location;
        result_td.data.normalized_filt_power = (filt_power - mean(filt_power,2))./std(filt_power);
        
        % absolute spectral power - units: uV^2/Hz
        % in this case, window size = 10 sec (non-overlapping)
        window_size = 10;
        time_bandwith_product = 2;
        number_tapers = 3;
        spectrum_window_size = 3; % in seconds
        step_size = 10; % in seconds
        bandpass = power_param.frequency_band;  
        [recording.data,recording.channels_location] = filter_non_scalp_recording(recording.data,recording.channels_location); %filter non-scalp chans
        result_sp = na_spectral_power(recording, window_size, time_bandwith_product, number_tapers, spectrum_window_size, bandpass,step_size);
        result_sp.metadata.channels_location = recording.channels_location;
        result_sp.metadata.number_channels = length(recording.channels_location);
        save(power_state_filename, 'result_sp','result_td');
        
        % Plot topographic distrubution
        figure
        topographic_map(result_td.data.normalized_filt_power,result_td.data.filt_location);
        title(strcat(participant," at ", state, " Power"))
        output_figure_path = strcat(power_participant_output_path,filesep,state,'_power.fig');
        savefig(output_figure_path)
        close(gcf)
        
         %Collect data for average, if applicable
        if power_param.average
            
            if(isstruct(avg_data.location) == 0)
                avg_data.location = recording.channels_location; %assumes first participant in average has all 129 channels
            end
            
            for e_i=1:length(avg_data.location)
                current_label = avg_data.location(e_i).labels;
                is_found = 0;
                for j=1:length(recording.channels_location)
                    if(strcmp(recording.channels_location(j).labels, current_label))
                        is_found = j;
                        break;
                    end
                end
                
                if(is_found ~= 0)
                    j = is_found;
                    avg_data.power_td(s,e_i) = avg_data.power_td(s, e_i) +  result_td.data.filt_power(j);
                    avg_data.power_count(s,e_i) = avg_data.power_count(s,e_i) + 1;
                end
            end
            
        end
        
        
    end
end

%Generate the average figure, if applicable
if power_param.average
    
    for s = 1:length(states)
        for c_i = 1:99
            avg_data.avg_power_td(s,c_i) = avg_data.power_td(s,c_i) ./ avg_data.power_count(c_i);
        end
    end
    
    
    figure
    for e_i = 1:length(states)
        avg_data.normalized_avg_power_td = (avg_data.avg_power_td(e_i,:)-mean(avg_data.avg_power_td(e_i,:),2))./std(avg_data.avg_power_td(e_i,:));
        subplot(ceil(length(states)/3),3,e_i)
        title(strcat("Power at ",states{e_i}))
        topographic_map(avg_data.normalized_avg_power_td,avg_data.location);
    end
    save(strcat(power_output_path,filesep,'_average_power.mat'), 'avg_data');
    output_figure_path = strcat(power_output_path,filesep,'_average_power.fig');
    savefig(output_figure_path)
    
end

function topographic_map(data,location)
topoplot(data,location,'maplimits','absmax', 'electrodes', 'off');
min_color = min(data);
max_color = max(data);
caxis([min_color max_color])
colormap('jet')
colorbar;
end