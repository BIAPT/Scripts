function [sifted_vector] = get_anterior(vector,channels_location)
%   GET_ANTERIOR getter function to fetch only the part of the vector that are
%   in the anterior part of the brain
%   vector: motif frequency count vector of length number of channels
%   channels_location: chanlocs data structure with channel information
%
%   sifted_vector: is the motif frequency count vector minus the channels
%   location who didn't meet the threshold.

    sifted_vector = []; 
    for i = 1:length(vector)
        
        % Every channels that are anterior to the center line of the
        % headset is defined as anterior
        if(channels_location(i).X > -0.001)
            sifted_vector = [sifted_vector, vector(i)];
        end
    end
end