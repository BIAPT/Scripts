function [sifted_vector, sifted_locations] = remove_channels(vector,channels_location,valid_channels_location)
%   REMOVE_CHANNELS processing function to remove channels from a vector
%   vector: is of length M
%   channels_location: is a struct of length M
%   valid_channels_location: is a struct of length N where N < M
%
%   sifted_vector: is of length N
%   sifted_location: is a struct of length N

    sifted_vector = [];
    sifted_locations = [];
    
    for i = 1:length(vector)
        label = channels_location(i).labels;
        
        % If the current label is valid we add both the data and the
        % location to their sifted version
        if(is_valid_channel(label,valid_channels_location))
           sifted_vector = [sifted_vector, vector(i)]; 
           sifted_locations = [sifted_locations, channels_location(i)];
        end
    end
end

% Helper function to check if a given label is part of the valid set
function [is_valid] = is_valid_channel(label,valid_channels_location)
    is_valid = 0;  
    for i = 1:length(valid_channels_location)
        if(strcmp(label,valid_channels_location(i).labels))
           is_valid = 1;
           break;
        end
    end
end

