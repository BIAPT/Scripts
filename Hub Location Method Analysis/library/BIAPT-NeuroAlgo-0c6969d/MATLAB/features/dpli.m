function [corrected_dpli] = dpli(eeg_data, number_surrogates, p_value)
%DPLI calculate weighted PLI and do some correction
%   Input:
%       eeg_data: data to calculate pli on
%       number_surrogates: number of surrogates dpli to create
%       p_value: p value to do the testing with the number of surrogates
%   Output:
%       corrected_dpli: dPLI with a correction 

    %% Seting up variables
    number_channels = size(eeg_data,1);
    surrogates_dpli = zeros(number_surrogates,number_channels,number_channels);
    eeg_data = eeg_data';
    
    %% Calculate dPLI
    uncorrected_dpli = directed_phase_lag_index(eeg_data); % uncorrected
    uncorrected_dpli(isnan(uncorrected_dpli)) = 0.5; %Have to do this otherwise NaN break the code
    
    %% Generate Surrogates
    parfor index = 1:number_surrogates
        surrogates_dpli(index,:,:) = directed_phase_lag_index_surrogate(eeg_data);
    end
    
    %% Correct the dPLI
    corrected_dpli = zeros(size(uncorrected_dpli));
    %Here we compare the calculated dPLI versus the surrogate
    %and test for significance

    %if the result is significant then 4 conditions are possible
    %1.dPLI value is greater than 0.5 and the median of the surrogate
    %is greater than 0.5
    %2.dPLI value is smaller than 0.5 and the median of the surrogate
    %is smaller than 0.5
    %3.dPLI is greater than 0.5 and median of surrogate is smaller
    %than 0.5
    %4.dPLI is smaller than 0.5 and median of surrogate is greater
    %than 0.5
    for m = 1:length(uncorrected_dpli)
        for n = 1:length(uncorrected_dpli)
            test = surrogates_dpli(:,m,n);
            p = signrank(test, uncorrected_dpli(m,n)); 
            if m == n
                corrected_dpli(m,n) = 0.5;
            elseif p < p_value % 4 Conditions 
                if uncorrected_dpli(m,n) > 0.5 && median(test) > 0.5
                    gap = uncorrected_dpli(m,n) - median(test);
                    if(gap < 0)
                        corrected_dpli(m,n) = 0.5; 
                    else
                        corrected_dpli(m,n) = gap + 0.5; %Gap is positive here
                    end  
                elseif uncorrected_dpli(m,n) < 0.5 && median(test) < 0.5 % CASE 2
                    gap = uncorrected_dpli(m,n) - median(test);
                    if(gap > 0)
                        corrected_dpli(m,n) = 0.5; 
                    else
                        corrected_dpli(m,n) = gap + 0.5; %Gap is negative here
                    end
                elseif uncorrected_dpli(m,n) > 0.5 && median(test) < 0.5 %CASE 3
                    extra = 0.5 - median(test);
                    corrected_dpli(m,n) = uncorrected_dpli(m,n) + extra;
                    % Here might be the problem
                elseif uncorrected_dpli(m,n) < 0.5 && median(test) > 0.5 %CASE 4
                    extra = median(test) - 0.5;
                    corrected_dpli(m,n) = uncorrected_dpli(m,n) - extra;
                    % Here also might be the problem
                end
                
                % Here we correct for out of bound behavior
                %{
                if(corrected_dpli(m,n) < 0)
                    corrected_dpli(m,n) = 0;
                elseif(corrected_dpli(m,n) > 1)
                   corrected_dpli(m,n) = 1; 
                end
                %}
            else
                corrected_dpli(m,n) = 0.5;
            end
            
            % Here we apply what happen in the upper triangle to the lower
            % triangle (this will ensure symmetry)
            %corrected_dpli(n,m) = 1 - corrected_dpli(m,n);
        end
    end
end

function pli = directed_phase_lag_index(data)
    % Given a multivariate data, returns phase lag index matrix
    % Modified the mfile of 'phase synchronization'
    % PLI(ch1, ch2) : 
    % if it is greater than 0.5, ch1->ch2
    % if it is less than 0.5, ch2->ch1

    number_channel = size(data, 2); % column should be channel

    %%%%%% Hilbert transform and computation of phases
    for i=1:number_channel
        segment = data(:,i);
        phase(:,i) = angle(hilbert(segment));
    end

    pli = ones(number_channel, number_channel);

    for channel_i = 1:number_channel
        for channel_j= 1 :number_channel
            %%%%%% phase lage index
            phase_difference = phase(:, channel_i) - phase(:, channel_j); % phase difference
            %PLI(ch1,ch2)=mean(sign(PDiff)); % only count the asymmetry
            pli(channel_i, channel_j) = mean(heaviside(sin(phase_difference)));
        end
    end
end

function surrogate_pli = directed_phase_lag_index_surrogate(data)
    % Given a multivariate data, returns phase lag index matrix
    % Modified the mfile of 'phase synchronization'
    % PLI(ch1, ch2) : 
    % if it is greater than 0.5, ch1->ch2
    % if it is less than 0.5, ch2->ch1

    number_channel=size(data,2); % column should be channel
    splice = randi(length(data));  % determines random place in signal where it will be spliced

    %%%%%% Hilbert transform and computation of phases
    for i=1:number_channel
        segment=data(:,i);
        %     phi0=angle(hilbert(x));  % only the phase component
        %     phi1(:,i)=unwrap(phi0);  % smoothing
        phase(:,i) = angle(hilbert(segment));
        random_phase(:,i) = [phase(splice:length(phase),i); phase(1:splice-1,i)];  % %This is the randomized signal
    end

    surrogate_pli = ones(number_channel,number_channel);

    for channel_i = 1:number_channel
        for channel_j = 1:number_channel
            %%%%%% phase lage index
            phase_difference=phase(:,channel_i)-random_phase(:,channel_j); % phase difference
    %         PLI(ch1,ch2)=mean(sign(PDiff)); % only count the asymmetry
            surrogate_pli(channel_i,channel_j)=mean(heaviside(sin(phase_difference)));
        end
    end
end