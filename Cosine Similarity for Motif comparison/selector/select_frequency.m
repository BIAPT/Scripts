function selected_data = select_frequency(data,frequency_labels)
%   SELECT_FREQUENCY Select the relevant frequency from the data
%   data: data to be select from
%
%   selected_data: data with only the relevant frequency selected

    % Selection dialog displaying
    [index,~] = listdlg('ListString',frequency_labels,'SelectionMode','single');
    label = frequency_labels(index);
    
    % Select the right subset of the data
    if(strcmp(label,"Alpha"))
        selected_data = data.alpha_frequency_avg.node_frequency;
    elseif(strcmp(label,"Theta"))
        selected_data = data.theta_frequency_avg.node_frequency;
    else
        error("Select at least a frequency.");
    end
end

