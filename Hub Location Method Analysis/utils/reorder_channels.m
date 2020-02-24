function [reorder_wpli] = reorder_channels(wpli)
%EEG_MANIPULATION is a wrapper function to call the python function
    
    % Save the original directory and move to the other path
    channel_order = readtable('biapt_egi129.csv');
    
    [num_location, labels, regions] = get_num_location(wpli.location, channel_order);
    
    reorder_wpli = struct();
    reorder_wpli.data = zeros(num_location, num_location);
    reorder_wpli.location = labels;
    reorder_wpli.regions = regions;
    
    for l1 = 1:length(labels)
       label_1 = labels{l1};
       
        for l2 = 1:length(labels)
            label_2 = labels{l2};
            
            index_1 = get_index_label(wpli.location, label_1);
            index_2 = get_index_label(wpli.location, label_2);
            
            if(index_1 == 0 || index_2 == 0)
               continue 
            end
            
            reorder_wpli.data(l1,l2) = wpli.data(index_1, index_2);
        end
    end
end


function [index] = get_index_label(location, target)
    
    index = 0;  
    for l = 1:length(location)
       label = location(l).labels;
       if(strcmp(label,target))
           index = l;
            return 
       end
    end
    
end

function [num_location, labels, regions] = get_num_location(location, total_location)
    num_location = 0;
    labels = {};
    regions = {};
    for i = 1:height(total_location)
       label = total_location(i,1).label{1};
       region = total_location(i,2).region{1};

       if(get_index_label(location, label) ~= 0)
           num_location = num_location + 1;
           labels{end+1} = label;
           regions{end+1} = region;
       end
    end
end
