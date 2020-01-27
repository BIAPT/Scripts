% Yacine Mahdid 2020-01-09
% This script was made for the revision of the motif analysis paper, where
% we want to compare alpha topographic map for each participant version the
% motif frequency topographic map

% Variables
states = {'BASELINE', 'IF5', 'EMF5','EML30','EML10','EML5','RECOVERY'};
extension = '.set';

participants = {'MDFA03','MDFA05','MDFA06','MDFA07','MDFA10','MDFA11','MDFA12','MDFA15','MDFA17'};
for p = 1:length(participants)
    participant = participants{p};
    input_path = strcat('E:/datasets/consciousness/MDFA Anesthesia data/',participant);
    output_path = strcat('E:/research projects/motif_analysis/power/',participant,'.mat');
    state_data = [];
    for i = 1:length(states)
       filename = strcat(participant,'_',states{i},extension);
       disp(filename)
       recording = load_set(filename,input_path);

       % Experiment Variables
        % Alpha Topographic Map Properties
        window_size = floor(recording.length_recording / recording.sampling_rate); % in seconds
        step_size = window_size; % in seconds
        bandpass = [8 13]; % in Hz
        result_td = na_topographic_distribution(recording, window_size, step_size, bandpass);
        [power, location] = filter_non_scalp(result_td.data.power, result_td.metadata.channels_location);

        data = struct();
        data.power = power;
        data.location = location;
        state_data = [state_data data];
    end

    %% Make the figure
    figure
    for e_i = 1:length(states)
        power = state_data(e_i).power;
        location = state_data(e_i).location;
        cur_title = strcat(participant," ",states{e_i});
        subplot(3,3,e_i)
        title(cur_title)
        topographic_map(power,location);
    end

    %% Save the data
    save(output_path, 'state_data');
end

function topographic_map(data,location)
    topoplot(data,location,'maplimits','absmax', 'electrodes', 'off');
    min_color = min(data);
    max_color = max(data);
    caxis([min_color max_color])
    colormap('jet')
    colorbar;
end