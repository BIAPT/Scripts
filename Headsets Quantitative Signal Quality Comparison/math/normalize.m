function [normalized_data] = normalize(raw_data)
%   NORMALIZE normalize the raw signal with the mean and standard deviation
%   raw_data: the unormalized_data
%   
%   normalized_data: the normalized data

    % Variable initialization
    number_motifs = size(raw_data,1);
    number_channels = size(raw_data,2);
    normalized_data = zeros(number_motifs,number_channels);
    
    % Normalize every motifs
    for i = 1:number_motifs
        if(std(raw_data(i,:)) ~= 0)
            normalized_data(i,:) = (raw_data(i,:) - mean(raw_data(i,:)))/std(raw_data(i,:));
        end
    end
end
