function[]=plot_circle(matrix,recording,threshold, mode)
    matrix = squeeze(matrix);

    % Create the figure
    figure('visible','on');
    colormap('jet');
    imagesc(matrix);
    colorbar;

    [x1,x2,y1,y2,circ_figure]=default_circle('on');
    
    %plot individual electrodes location
    el_data=get_electrode_circle_locations(recording,x1,y1,x2,y2);

    maxvalue=max(matrix,[],[1,2]);
    plot_connectivity_line(matrix,el_data,maxvalue, threshold, mode)
    
    for i=1:height(el_data(:,1))
        plot_points_labels(mode,string(el_data.name(i)),string(el_data.region(i)),...
            cell2mat(el_data.X_location(i)),cell2mat(el_data.y_location(i)),...
            cell2mat(el_data.X_label(i)),cell2mat(el_data.y_label(i)))
    end
    
    
end
