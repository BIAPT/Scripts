function plot_motif(frequency, source, distance, channels_location, title_name)
    
    % Will plot motif 
    figure;
    ax1 = subplot(1,3,1);
    title(title_name);
    topoplot(frequency,channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    ax2 = subplot(1,3,2);
    title(title_name);
    topoplot(distance,channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    ax3 = subplot(1,3,3);
    title(title_name);
    topoplot(source,channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    
    colormap(ax1,'jet');
    colormap(ax2,'bone');
    colormap(ax3,'hot');
end