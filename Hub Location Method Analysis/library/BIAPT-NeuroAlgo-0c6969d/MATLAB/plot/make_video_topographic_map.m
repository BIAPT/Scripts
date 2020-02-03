function make_video_topographic_map(filename, topographic_map, channels_location, step_size)
%make_video_functional_connectivity Summary of this function goes here
%   Detailed explanation goes here
    [number_frames, number_channels] = size(topographic_map);
    all_edges = reshape(topographic_map, [1 (number_frames*number_channels)]);
    min_color = min(all_edges);
    max_color = max(all_edges);
    
    video = VideoWriter(strcat(filename,'.avi')); %create the video object
    video.FrameRate = 1/step_size;
    open(video); %open the file for writing
    for i=1:number_frames %where N is the number of images
        disp(strcat("Making video: ", string(i/number_frames)," %"));
        vector = squeeze(topographic_map(i,:));
        
        % Create the figure
        fc_figure = figure('visible','off');
        topoplot(vector,channels_location,'maplimits','absmax', 'electrodes', 'off');

        colormap('jet');
        colorbar;
        caxis([min_color max_color])
        writeVideo(video,getframe(fc_figure)); %write the image to file
        delete(fc_figure)
    end
    close(video); %close the file
end