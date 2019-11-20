function plot_weights_bar(type, weights)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    % Make a figure with subplot
    figure
    subplot(2,1,1)
    bar(weights(1:84))
    title(strcat(type, ' Weights of Mean'))
    xlim([0,83])
    ylim([-1.6,1.8])

    subplot(2,1,2)
    bar(weights(84:end))
    title(strcat(type, ' Weights of Std'))
    xlim([0,83])
    ylim([-1.6,1.8])
end

