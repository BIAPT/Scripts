function [data,data_pli,data_aec,label,identity] = get_both_data(epoch,frequency,regions_mask,folder)
    % Create a filepath using the epoch and frequency
    aec_file_name = strcat(folder,filesep,"aec_",epoch,"_",frequency,".mat");
    pli_file_name = strcat(folder,filesep,"pli_",epoch,"_",frequency,".mat");
    % Load the data
    aec_data = load(aec_file_name);
    pli_data = load(pli_file_name);
    pli_data = pli_data.PLI_OUT;
    aec_data = aec_data.AEC_OUT;

    % Variable Initialization
    number_participant = length(aec_data);
    
    % Fix aec to match pli in orientation
    for i = 1:number_participant
       aec_data{i} = permute(aec_data{i},[3 2 1]);
    end
    
    % Match the two dataset
    for i = 1:number_participant
        pli_window_length = size(pli_data{i},1);
        aec_window_length = size(aec_data{i},1);
        
        min_window_length = min([pli_window_length aec_window_length]);
        pli_data{i} = pli_data{i}(1:min_window_length,:,:);
        aec_data{i} = aec_data{i}(1:min_window_length,:,:);
    end

    % Process the remaining data and merge theminto one file
    data_pli = [];
    data_aec = [];
    merged_data = [];
    for i = 1:number_participant
       flatten_data_pli = process_data(pli_data{i},regions_mask); 
       flatten_data_aec = process_data(aec_data{i},regions_mask);
       processed_data_both = cat(2,flatten_data_aec,flatten_data_pli);
       merged_data = [merged_data; processed_data_both];
       
       data_pli = [data_pli; flatten_data_pli];
       data_aec = [data_aec; flatten_data_aec];
       
       if(i == 1)
          identity = repelem(1,size(processed_data_both,1)); 
       else
           current_identity = repelem(i,size(processed_data_both,1));
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
    elseif(strcmp(epoch,"ec8"))
        number_label = 5;
    end
    
    label = repelem(number_label,size(merged_data,1));
    data = merged_data;
end

