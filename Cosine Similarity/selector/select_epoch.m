function selected_data = select_epoch(data,list)
%SELECT_EPOCH Summary of this function goes here
%   Detailed explanation goes here
    [index,~] = listdlg('ListString',list,'SelectionMode','single');
    
    
    label = list(index);
    
    selected_data = [];
    for i=1:length(data)
        if(strcmp(label,data(i).name))
           selected_data = data(i);
           return
        end
    end
    
    error("Select at least an epoch label.");
end

