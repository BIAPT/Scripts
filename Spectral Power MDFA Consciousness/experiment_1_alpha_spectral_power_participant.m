% Yacine Maahdid 2020-01-09
% This script was made for the revision of the motif analysis paper, where
% we want to compare alpha topographic map for each participant version the
% motif frequency topographic map

% Variables
participant = 'MDFA17';
states = {'BASELINE','EMF5','EML5','EML10','EML30','IF5','RECOVERY'};
extension = '.set';

filename = 'MDFA17_BASELINE.set'; 
filepath = 'C:\Users\biapt\Desktop\motif fix\mdfa17_data';

state_data = [];
for i = 1:length(states)
   filename = strcat(participant,'_',states{i},extension);
   disp(filename)
   recording = load_set(filename,filepath);
   
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

% Make the figure
figure
for i = 1:length(states)
    power = state_data(i).power;
    location = state_data(i).location;
    cur_title = strcat(participant," ",states{i});
    subplot(3,3,i)
    title(cur_title)
    topographic_map(power,location);
end

function topographic_map(data,location)
    topoplot(data,location,'maplimits','absmax', 'electrodes', 'off');
    min_color = min(data);
    max_color = max(data);
    caxis([min_color max_color])
    colormap('jet')
    colorbar;
end