function [merged_data,label,identity] = get_data(data,type,regions_mask,epoch)
    % Get the data
    if(strcmp(type,"pli"))
       data = data.PLI_OUT;
        % Getting the data out of the cells and concatenating them together
        merged_data = process_data(data{1},regions_mask);
        identity = repelem(1,size(merged_data,1));
        for i = 2:length(data)
            processed_data = process_data(data{i},regions_mask);
            merged_data = cat(1,merged_data,processed_data);
            current_identity = repelem(i,size(processed_data,1));
            identity = cat(2,identity,current_identity);
        end
    elseif(strcmp(type,"aec"))
        data = data.AEC_OUT;
        merged_data = process_data(permute(data{1},[3 2 1]),regions_mask);
        identity = repelem(1,size(merged_data,1));
        for i = 2:length(data)
            processed_data = process_data(permute(data{i},[3 2 1]),regions_mask);
            merged_data = cat(1,merged_data,processed_data);
            current_identity = repelem(i,size(processed_data,1));
            identity = cat(2,identity,current_identity);
        end     
    end
    
    % Make the label
    if(strcmp(epoch,"ec1"))
        number_label = 0;
    elseif(strcmp(epoch,"if5"))
        number_label = 1;
    elseif(strcmp(epoch,"emf5"))
        number_label = 2;
    elseif(strcmp(epoch,"eml5"))
        number_label = 3;
    elseif(strcmp(epoch,"ec3"))
        number_label = 4;
    end
    label = repelem(number_label,size(merged_data,1));
end