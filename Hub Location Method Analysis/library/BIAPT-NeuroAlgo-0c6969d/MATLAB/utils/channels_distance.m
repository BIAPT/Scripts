function [distance] = channels_distance(location_1, location_2)
%CHANNELS_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

    % here we are using the euclidean distance
    % location_1_2 are vecor of [X,Y,Z]
    distance = sqrt(sum((location_1 - location_2) .^ 2));
end

