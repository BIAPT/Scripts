function [selected_data, other_selected_data] = select_motif(data,other_data,list)
%SELECT_EPOCH Summary of this function goes here
%   Detailed explanation goes here
    [index,~] = listdlg('ListString',list,'SelectionMode','single');
    
    if(isempty(index) == 0)
        selected_data = data(index,:);
        other_selected_data = other_data(index,:);
    else
        error("Select one motif.");
    end
end

