function [norm_frequency] = normalize_motif(frequency)
%NORMALIZE_MOTIF Summary of this function goes here
%   Detailed explanation goes here

    [num_motifs,~] = size(frequency);
    norm_frequency = frequency;
    for i = 1:num_motifs  
        if(std(frequency(i,:)) ~= 0)
            norm_frequency(i,:) = (frequency(i,:) - mean(frequency(i,:)))/std(frequency(i,:));    
        end    
    end
end

