function make_video_hub_location(filename, degree_map, median_location, max_location, channels_location, step_size)
%make_video_functional_connectivity Summary of this function goes here
%   Detailed explanation goes here
    [number_frames, number_channels] = size(degree_map);
    all_edges = reshape(degree_map, [1 (number_frames*number_channels)]);
    min_color = min(all_edges);
    max_color = max(all_edges);
    
    video = VideoWriter(strcat(filename,'.avi')); %create the video object
    video.FrameRate = 1/step_size;
    open(video); %open the file for writing
    for i=1:number_frames %where N is the number of images
        disp(strcat("Making video: ", string(i/number_frames)," %"));
        channels_degree = squeeze(degree_map(i,:));
        
        % Create the figure
        fc_figure = plot_hub( channels_degree, median_location(i), max_location(i), channels_location, min_color, max_color);
        writeVideo(video,getframe(fc_figure)); %write the image to file
        delete(fc_figure)
    end
    close(video); %close the file
end

function [figure_handle] = plot_hub( channels_degree, median_location, max_location, channels_location, min_color, max_color)
    % Plotting the hub location
    figure_handle = figure('visible','off');
    subplot(1,3,1)
    topoplot(channels_degree,channels_location,'maplimits','absmax', 'electrodes', 'off');
    colormap('cool');
    colorbar;
    caxis([min_color max_color])
    title('Degrees')
    subplot(1,3,2);
    bar(median_location);
    ylim([0 1])
    title('Median Location');
    subplot(1,3,3);
    bar(max_location);
    ylim([0 1])
    title('Max Location');
end