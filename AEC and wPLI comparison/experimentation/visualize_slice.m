video = VideoWriter('participant_1_aec.avi'); %create the video object
video.FrameRate = 5;
open(video); %open the file for writing
for i = 1:59
    aec = squeeze(participant_1_aec(:,:,i));
    aec = normalize_data(aec);
    %pli_plot = figure;
    %colormap('jet')
    %imagesc(aec);
    %colorbar();
   

    writeVideo(video,aec); %write the image to file
end
close(video); %close the file