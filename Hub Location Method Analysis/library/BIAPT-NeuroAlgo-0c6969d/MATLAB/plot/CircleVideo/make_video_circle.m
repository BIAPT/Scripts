function make_video_circle(filename, functional_connectivity_matrices,recording,threshold, mode, step_size)
%make_video_functional_connectivity Summary of this function goes here
%   Detailed explanation goes here
    [number_frames, number_channels, ~] = size(functional_connectivity_matrices);
    all_edges = reshape(functional_connectivity_matrices, [1 (number_frames*number_channels*number_channels)]);
    maxvalue = max(all_edges);
    
    video = VideoWriter(strcat(filename,'.avi')); %create the video object
    video.FrameRate = 1/step_size;
    open(video); %open the file for writing
    
    %plot default circle
    [x1,x2,y1,y2,circle_figure]=default_circle('off');
    
    %plot individual electrodes location
    el_data=get_electrode_circle_locations(recording,x1,y1,x2,y2);

    
    for i=1:height(el_data(:,1))
        plot_points_labels(mode,string(el_data.name(i)),string(el_data.region(i)),...
            cell2mat(el_data.X_location(i)),cell2mat(el_data.y_location(i)),...
            cell2mat(el_data.X_label(i)),cell2mat(el_data.y_label(i)))
    end
    
        
    for i=1:number_frames %where N is the number of images
        disp(strcat("Making video: ", string((i/number_frames)*100)," %"));
        frame_figure = copyobj(circle_figure,0);
        matrix = squeeze(functional_connectivity_matrices(i,:,:));
        plot_connectivity_line(matrix,el_data,maxvalue, threshold, mode)
        writeVideo(video,getframe(frame_figure)); %write the image to file
        delete(frame_figure)     
    end
    close(video); %close the file
end


