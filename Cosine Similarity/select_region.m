function [selected_data, other_selected_data] = select_region(data,other_data,channels_location,list)
%SELECT_EPOCH Summary of this function goes here
%   Detailed explanation goes here
    [index,~] = listdlg('ListString',list,'SelectionMode','single');
    label = list(index);
    if(strcmp(label,'Anterior'))
        selected_data = get_anterior(data,channels_location);
        other_selected_data = get_anterior(other_data,channels_location);
    elseif(strcmp(label,'Posterior'))
        selected_data = get_posterior(data,channels_location);
        other_selected_data = get_posterior(other_data,channels_location);
    else
        error("Select one region.");
    end
end

