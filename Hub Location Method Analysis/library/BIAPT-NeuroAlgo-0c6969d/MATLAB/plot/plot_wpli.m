function  plot_wpli(data, title_name, label_names, color, isInterpolation)
%PLOT_wPLI Summary of this function goes here
%   Detailed explanation goes here
    
    figure;
    % we are using imagesc for colored matrix image without interpolation
    % and pcolor for image with interpolation
    if(isInterpolation)
        image = pcolor(data);
        image.FaceColor = 'interp';
    else
        imagesc(data);
    end
    
    % setting the x ticks to be vertical with the right label names
    xtickangle(90)
    xticklabels(label_names);
    xticks(1:length(label_names));
    
    % setting the y ticks with the right label names
    yticklabels(label_names); 
    yticks(1:length(label_names));
    
    % add-in the title
    title(title_name);
    
    % Setting the colorbar
    colorbar;
    colormap(color);
    
end