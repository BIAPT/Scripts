function [sifted_vector] = get_posterior(vector,channels_location)
%   GET_POSTERIOR getter function to fetch only the part of the vector that are
%   in the posterior part of the brain
%   vector: motif frequency count vector of length number of channels
%   channels_location: chanlocs data structure with channel information
%
%   sifted_vector: is the motif frequency count vector minus the channels
%   location who didn't meet the threshold.

    sifted_vector = [];
    for i = 1:length(vector)
        
        % Every channels that are in the below the center line of the
        % headset is defined as posterior
        if(channels_location(i).X < 0.001)
            sifted_vector = [sifted_vector, vector(i)];
        end
    end
end