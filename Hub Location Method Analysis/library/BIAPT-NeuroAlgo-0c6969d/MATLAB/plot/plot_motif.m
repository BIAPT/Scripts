function plot_motif(frequency, title_name, channels_location, color)
%PLOT_MOTIF Summary of this function goes here
%   The frequency here is for a single motif
    figure;
    title(title_name);
    topoplot(frequency,channels_location,'maplimits','absmax', 'electrodes', 'off');
    colorbar;
    colormap(color);
end

