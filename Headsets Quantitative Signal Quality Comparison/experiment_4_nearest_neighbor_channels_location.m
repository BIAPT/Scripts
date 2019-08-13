%% Yacine Mahdid 2019-08-13
% This script was written in the context of a comment we got for my paper
% on headsets comparison of signal quality using functional connectivity.
% The reviewer wanted to see a number associated to how similar to gold
% standard the other headset features were. In this script I will develop a
% nearest neighbor based approach to map one headset into another.

%% Load the two headset to compare
path_egi = "C:\Users\Fabien Dal Maso\Documents\eeg_headset_comparison\_revision\exported_data\egi\egi_closed_1.mat";
path_dsi = "C:\Users\Fabien Dal Maso\Documents\eeg_headset_comparison\_revision\exported_data\dsi\dsi_closed_1.mat";

egi_location = load_headset_location(path_egi);
dsi_location = load_headset_location(path_dsi);


%% Use one headset as a filter for the other headset location
filtered_egi_location = filter_location(egi_location,dsi_location);

%% Visualize the two headset channels that are left
figure;
plot_eeg_location(egi_location,'blue','o');
hold on;
plot_eeg_location(dsi_location,'red','o');
hold on;
plot_eeg_location(filtered_egi_location,'green','*');

%% Helper function
% will load the headset location given a path
function [eeg_location] = load_headset_location(path)
    data = load(path);
    eeg_location = data.EEG.chanlocs;
end

% Will scan through each of the location filter to find its nearest
% neighbor in the eeg location to be filtered. It will output a filtered
% version of te eeg location.
function [filtered_eeg_location] = filter_location(eeg_location, filter)
    
    % Create the location 3D matrix (X,Y,Z) for each electrodes
    location = zeros(length(eeg_location),3);
    for i = 1:length(eeg_location)
        radius = eeg_location(i).sph_radius;
        location(i,1) = eeg_location(i).X/radius;
        location(i,2) = eeg_location(i).Y/radius;
        location(i,3) = eeg_location(i).Z/radius;
    end

    % Find the optimal location
    closest_location = zeros(1,length(filter));
    for i = 1:length(filter)
        radius = filter(i).sph_radius;
        spatial_filter = [filter(i).X filter(i).Y filter(i).Z]/radius;
        closest_location(i) = knnsearch(location,spatial_filter,'K',1,'Distance','euclidean');
    end
    
    % Filter based on the index
    filtered_eeg_location = eeg_location(closest_location);
end

function plot_eeg_location(location, color,type)
    X = zeros(1,length(location));
    Y = zeros(1,length(location));
    Z = zeros(1,length(location));
    
    for i = 1:length(location)
        radius = location(i).sph_radius;
        X(i) = location(i).X/radius;
        Y(i) = location(i).Y/radius;
        Z(i) = location(i).Z/radius;
    end
    
    scatter3(X,Y,Z,type,'MarkerFaceColor',color)
    
end