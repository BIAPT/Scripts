function [temp_sqi] = signal_quality_index_temp(temperature)
% Function used to compute a data quality index for a measurement session
            
% Input: filename - including path if not in local directory.

    % Clean existing variables in function workspace so it does not affect
    % result consistency
    clear_internal;

    % Skin Temperature signal 
    temperature = medfilt1(temperature,1);

    %% Signal Quality Index Calculation
    % Setup variables
    temp_sqi = ones(length(temperature),1);

    % Skin temperature SQI 
    W = 50;
    L = length(temperature);  
    for l1 = 1:25:L-W+1 %value betweeen 1 and L is half the window size   
        l2 = l1 - 1+W;
        dff = std(temperature(l1:l2));
        temp_min = min(temperature(l1:l2));
        temp_maxmin = max(temperature(l1:l2)) - min(temperature(l1:l2));
        temp_sqi(l1:l2) = temp_sqi(l1:l2)*exp(-0.20*dff); %sensitivity to SD
        
        %identify out of range values
        if temp_min < 15  
                temp_sqi(l1:l2) = 0.50*temp_sqi(l1:l2); 
        end
        
        %identify flat lines 
        if max(abs(diff(temperature(l1:l2)))) <= 0.0001 
                temp_sqi(l1:l2) = 0.50*temp_sqi(l1:l2);
        end    
        
        %identify steep increases/decreases
        if abs(temp_maxmin) > 4 
            temp_sqi(l1:l2) = 0.3*temp_sqi(l1:l2);
        end
    end

    if mod(L, 2) ~=0 && L>50
        l1 = L-W;
        l2 = L;
        dff = std(temperature(l1:l2));
        temp_min = min(temperature(l1:l2));
        temp_maxmin = max(temperature(l1:l2)) - min(temperature(l1:l2));
        temp_sqi(l1:l2) = temp_sqi(l1:l2)*exp(-0.20*dff);

        if temp_min < 15
            temp_sqi(l1:l2) = 0.50*temp_sqi(l1:l2);
        end
        
        if max(abs(diff(temperature(l1:l2)))) < 0.0001
            temp_sqi(l1:l2) = 0.50*temp_sqi(l1:l2);
        end     
        
        if abs(temp_maxmin) > 4
            temp_sqi(l1:l2) = 0.3*temp_sqi(l1:l2);
        end
    end

    %% 4 - Make a decision based on the three signal quality indexes 

    for j = 1:length(temp_sqi)
        if temp_sqi(j) <= 0.5
            weight_skt(j) = 2;
        else
            weight_skt(j) = 1;
        end
    end

    temp_sqi = temp_sqi.*weight_skt';
end
