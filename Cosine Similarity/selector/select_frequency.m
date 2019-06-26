function selected_data = select_frequency(data,list)
%SELECT_EPOCH Summary of this function goes here
%   Detailed explanation goes here
    [index,~] = listdlg('ListString',list,'SelectionMode','single');
    label = list(index);
    
    % Select the right subset of the data
    if(strcmp(label,"Alpha"))
        selected_data = data.alpha_frequency_avg.node_frequency;
    elseif(strcmp(label,"Theta"))
        selected_data = data.theta_frequency_avg.node_frequency;
    else
        error("Select at least a frequency.");
    end
end

