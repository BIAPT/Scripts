function  [avg_channels_location] = fit_channels_location(avg_channels_location, channels_location)
%FILTER_NON_SCALP Summary of this function goes here
%   Detailed explanation goes here

    for i=1:length(avg_channels_location)
        if(i > length(avg_channels_location))
            return
        end
        current_label = avg_channels_location(i);
        is_found = 0;
        for j=1:length(channels_location)
           if(strcmp(channels_location(j).labels,current_label))
               is_found = 1;
               break;
           end
        end
        
        if(is_found == 0)
            avg_channels_location(i) = [];
            i = i - 1;
        end
    end
end