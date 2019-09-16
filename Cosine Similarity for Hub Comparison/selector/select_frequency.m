function selected_data = select_frequency(frequency_labels)
%   SELECT_FREQUENCY Select the relevant frequency from the data
%   data_path: path to the data to be select from
%
%   selected_data: data for the relevant frequency selected

    % Selection dialog displaying
    [index,~] = listdlg('ListString',frequency_labels,'SelectionMode','single');
    label = frequency_labels(index);
    
    % Select the right subset of the data
    if(strcmp(label,"Alpha"))
        load average_degrees_pre_alpha
        selected_data = result_pre;
    elseif(strcmp(label,"Theta"))
        load average_degrees_pre_theta
        selected_data = result_pre;
    elseif(strcmp(label,"Delta"))
        load average_degrees_pre_delta
        selected_data = result_pre;
    else
        error("Select at least one frequency.");
    end
end

