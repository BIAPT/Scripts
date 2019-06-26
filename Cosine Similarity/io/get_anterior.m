function [sifted_vector] = get_anterior(vector,channels_location)
%GET_ANTERIOR Summary of this function goes here
%   Detailed explanation goes here

    sifted_vector = []; %% TODO check how many are smaller than -0.001
    for i = 1:length(vector)
        if(channels_location(i).X > -0.001)
            sifted_vector = [sifted_vector, vector(i)];
        end
    end
end