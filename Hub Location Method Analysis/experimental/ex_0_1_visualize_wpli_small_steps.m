%{
    Yacine Mahdid 2020-02-6
    Here we visualize what we just created in ex_0_0 in order to check if
    the matrices make sense. We are looking only at one pre configured
    state for one pre configured participant.
%}


% Setup the experiment
setup_experiments;

% Extract Needed Variables
ppt = settings.participant;
state = settings.state;
in_path = settings.output_path;
out_path = settings.output_path;

% Constructing the in and out filename
in_filename = strcat(in_path,ppt, '_', state, '_wpli.mat');
out_figure_path = mkdir_if_not_exist(in_path, 'figure');

% Load the wpli struct
data = load(in_filename);
result_wpli = data.result_wpli;

%% Make the figures for whole brain

% Plot the average wPLI matrix
out_avg_figure = strcat(out_figure_path,filesep, ppt,'_',state,'_avg_wpli.fig');
avg_wpli = result_wpli.data.avg_wpli;
title_name = strcat(ppt,' ',state,' Average wPLI');
label_names = '';
color = 'jet';
isInterpolation = 0;

hemisphere = 'right';
[avg_wpli, location] = get_hemisphere(avg_wpli, result_wpli.metadata.channels_location, hemisphere);

wpli = struct();
wpli.data = avg_wpli;
wpli.location = location;
avg_reorder_wpli = reorder_channels(wpli);

plot_wpli(avg_reorder_wpli.data, title_name, avg_reorder_wpli.regions, color, isInterpolation)

% Aggregate statistic over the whole wPLI matrices (mean and std)
out_stat_figure = strcat(out_figure_path, filesep, ppt,'_',state,'_stat_wpli.fig');
mean_wpli = mean(result_wpli.data.wpli,3);
std_wpli = std(result_wpli.data.wpli,1,3);

figure
subplot(2,1,1)
plot(mean_wpli)
title(strcat(ppt,' ', state, ' Global wPLI over time'))
subplot(2,1,2)
plot(std_wpli)
title(strcat(ppt,' ', state, ' Standard Deviation of Global wPLI over time'))

[num_wpli, num_channels, ~] = size(result_wpli.data.wpli);
location = result_wpli.metadata.channels_location;
reorder_wplis = zeros(num_wpli, 43, 43);

for i = 1:num_wpli
    disp(i)
    [half_wpli, half_location] = get_hemisphere(squeeze(result_wpli.data.wpli(i,:,:)), result_wpli.metadata.channels_location, hemisphere);
    wpli = struct();
    wpli.data = half_wpli;
    wpli.location = half_location;  
    
    reorder_wpli = reorder_channels(wpli);
    reorder_wplis(i,:,:) = reorder_wpli.data;
end
% Generate a video of the whole wPLI matrices over time
out_video = strcat(hemisphere);
make_video_functional_connectivity(out_video, reorder_wplis, .1)
 
function [half_wpli, half_location] = get_hemisphere(wpli, location, hemisphere)
    
    half_channels = [];
    for i = 1:length(location)
       if(is_hemisphere(location, i, hemisphere))
          half_channels = [half_channels, i]; 
       end
    end
    
    num_channel = length(half_channels);
    half_wpli = zeros(num_channel, num_channel);
    half_location = location(half_channels);
    for i = 1:num_channel
        ci = half_channels(i);
        
        for j = 1:num_channel
            cj = half_channels(j);
            
            half_wpli(i,j) = wpli(ci, cj);
        end
        
    end

end

function is = is_hemisphere(location, index, hemisphere)
    is = 0;
    if(strcmp(hemisphere, 'left') && location(index).is_left)
        is = 1;
    elseif(strcmp(hemisphere, 'right') && location(index).is_right)
       is = 1; 
    end
end