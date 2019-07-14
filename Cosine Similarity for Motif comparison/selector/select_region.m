function [selected_data, other_selected_data] = select_region(data,other_data,channels_location,region_labels)
%   SELECT_REGION selection dialog to select the region of interest from
%   the data
%   data: data of interest to select the region from
%   other_data: data of interest to select the region from
%   channels_location: eeg channels location for both dataset
%
%   selected_data: subset of data where only the region of interest where
%   selected
%   other_selected_data: subset of other data where only the region of
%   interest where selected.

    % Selection dialog
    [index,~] = listdlg('ListString',region_labels,'SelectionMode','single');
    label = region_labels(index);
    
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

