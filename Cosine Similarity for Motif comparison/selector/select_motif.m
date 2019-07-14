function [selected_data, other_selected_data] = select_motif(data,other_data,motif_labels)
%   SELECT_MOTIF Select only the relevant motif from the data
%   data: data to select from
%   other_data: data to select from
%
%   selected_data: selected data with only the motif of interest
%   other_selected_data: selected datawith only the motif of interest

    % Selection dialog displaying
    [index,~] = listdlg('ListString',motif_labels,'SelectionMode','single');
    
    if(isempty(index) == 0)
        selected_data = data(index,:);
        other_selected_data = other_data(index,:);
    else
        error("Select one motif.");
    end
end

