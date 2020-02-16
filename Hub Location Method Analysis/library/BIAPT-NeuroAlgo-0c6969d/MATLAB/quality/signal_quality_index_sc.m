function [ sc_sqi] = signal_quality_index_sc(skin_conductance)
%% function data_in = comp_SQI(data_in)
% Function used to compute a data quality index for a measurement session
            
% Input: filename - including path if not in local directory.

    % Clean existing variables in function workspace so it does not affect
    % result consistency
    clear_internal;
    
    %% Filtering the signals
    % Skin conductance signal
    
    % Median filtering
    skin_conductance = medfilt1(skin_conductance,75);
    
    % Derivative correction
    derr = diff(skin_conductance); 
    z_der = zscore(derr);
    for k=11:length(derr)-10
        % correct for steep positive slopes. 
        if z_der(k) >5 
            derr(k)= median(derr(k-10:k+10));
        end
        
        % and correct for steep negative slopes. value different than positive
        % threshold. More lenient in steep decreases than increases
        if z_der(k)<-3 
           derr(k)= median(derr(k-10:k+10));
        end
    end
    
    filt_derr = cat(1,skin_conductance(1), derr);
    skin_conductance = cumsum(filt_derr);

    %% Signal Quality Index Calculation
    % Setup variables
    sc_sqi = ones(length(skin_conductance),1);
    
    % SQI for Skin Conductance
    W = 30;
    L = length(skin_conductance);  
    for l1 = 1:15:L-W+1
        l2 = l1 - 1+W;
        dff = std(skin_conductance(l1:l2));
        sc_sqi(l1:l2) = sc_sqi(l1:l2)*exp(-0.1*dff); 
        
        if max(diff(skin_conductance(l1:l2))) > 3 
            sc_sqi(l1:l2) = 0.4*sc_sqi(l1:l2);
        end  
    end
    
    %the below code must match any changes made above.
    if mod(L, 2)~=0
       l1 = L-W;
       l2 = L;
       dff = std(skin_conductance(l1:l2));
       sc_sqi(l1:l2) = sc_sqi(l1:l2)*exp(-0.1*dff);
        if max(diff(skin_conductance(l1:l2))) > 3
            sc_sqi(l1:l2) = 0.4*sc_sqi(l1:l2);
        end  
    end

    W = 120; %identifiy flat lines and decrease their signal quality
    for l1 = 1:30:L-W+1 
        l2 = l1-1+W;
        if max(abs(diff(skin_conductance(l1:l2)))) <= 0.001
            sc_sqi(l1:l2) = 0.1*sc_sqi(l1:l2);
        end 
    end

    for j = 1:length(sc_sqi) %identify values out of normal range 
        if skin_conductance(j) <= 0.02
            sc_sqi(j) = 0;
        end
        
        if skin_conductance(j) > 20
           sc_sqi(j) = 0.65;
        end
         
        if skin_conductance(j) > 30
           sc_sqi(j) = 0;
        end
    end

    %% 4 - Make a decision based on the three signal quality indexes 

    for j=1:length(sc_sqi)
        if sc_sqi(j) <= 0.5
            weight_sc(j) = 2;
        else
            weight_sc(j) = 1;
        end
    end

    sc_sqi = sc_sqi.*weight_sc';
end
