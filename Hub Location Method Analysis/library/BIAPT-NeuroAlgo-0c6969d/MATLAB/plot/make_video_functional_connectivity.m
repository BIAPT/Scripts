function make_video_functional_connectivity(filename, functional_connectivity_matrices, step_size)
%make_video_functional_connectivity Summary of this function goes here
%   Detailed explanation goes here
    [number_frames, number_channels, ~] = size(functional_connectivity_matrices);
    all_edges = reshape(functional_connectivity_matrices, [1 (number_frames*number_channels*number_channels)]);
    min_color = min(all_edges);
    max_color = max(all_edges);
    
    video = VideoWriter(strcat(filename,'.avi')); %create the video object
    video.FrameRate = 1/step_size;
    open(video); %open the file for writing
    for i=1:number_frames %where N is the number of images
        disp(strcat("Making video: ", string(i/number_frames)," %"));
        matrix = squeeze(functional_connectivity_matrices(i,:,:));
        
        % Create the figure
        fc_figure = figure('visible','off');
        colormap('jet');
        imagesc(matrix);
        colorbar;
        caxis([min_color max_color])
        
        writeVideo(video,getframe(fc_figure)); %write the image to file
        delete(fc_figure)
    end
    close(video); %close the file
end


