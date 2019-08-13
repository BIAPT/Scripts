function selected_data = select_epoch(data,epoch_labels)
%   SELECT_EPOCH show a dialogue to select from the data only the relevant
%   epoch
%   data: is the data structure to select the epoch from
%
%   selected_data: is the relevant epoch of the data

    % Selection dialog
    [index,~] = listdlg('ListString',epoch_labels,'SelectionMode','single');
    label = epoch_labels(index);
    
    selected_data = [];
    for i=1:length(data)
        if(strcmp(label,data(i).name))
           selected_data = data(i);
           return
        end
    end
    
    error("Select at least an epoch label.");
end

