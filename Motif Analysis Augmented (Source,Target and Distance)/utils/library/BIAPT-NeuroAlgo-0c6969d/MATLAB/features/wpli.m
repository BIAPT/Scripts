function [corrected_wpli] = wpli(eeg_data, number_surrogates, p_value)
%WPLI calculate weighted PLI and do some correction
%   Input:
%       eeg_data: data to calculate pli on
%       eeg_info: info about the headset
%       number_surrogates: number of surrogates data to generate
%       p_value: p value level to test at
%   Output:
%       corrected_wpli: PLI with a correction (either p value or
%       substraction)

%% Seting up variables
    number_channels = size(eeg_data,1);
    surrogates_wpli = zeros(number_surrogates,number_channels,number_channels);
    eeg_data = eeg_data';
    
    %% Calculate wPLI
    uncorrected_wpli = weighted_phase_lag_index(eeg_data); % uncorrected
    uncorrected_wpli(isnan(uncorrected_wpli)) = 0; %Have to do this otherwise NaN break the code
    
    %% Generate Surrogates
    parfor index = 1:number_surrogates
        surrogates_wpli(index,:,:) = weighted_phase_lag_index_surrogate(eeg_data);
    end
    
    %% Correct the wPLI (either by substracting or doing a p test)
    %Here we compare the calculated dPLI versus the surrogate
    %and test for significance
    corrected_wpli = zeros(size(uncorrected_wpli));
    for m = 1:length(uncorrected_wpli)
        for n = 1:length(uncorrected_wpli)
            test = surrogates_wpli(:,m,n);
            p = signrank(test, uncorrected_wpli(m,n));       
            if p < p_value
                if uncorrected_wpli(m,n) - median(test) > 0 %Special case to make sure no PLI is below 0
                    corrected_wpli(m,n) = uncorrected_wpli(m,n) - median(test);
                end
            end          
        end
    end
end

function pli = weighted_phase_lag_index(data)
    number_channel = size(data,2); 
    a_sig = hilbert(data);
    pli = ones(number_channel,number_channel);

    for channel_i=1:number_channel-1
        for channel_j=channel_i+1:number_channel
            c_sig=a_sig(:,channel_i).*conj(a_sig(:,channel_j));

            numerator=abs(mean(imag(c_sig))); % average of imaginary
            denominator=mean(abs(imag(c_sig))); % average of abs of imaginary

            pli(channel_i,channel_j) = numerator/denominator;
            pli(channel_j,channel_i) = pli(channel_i,channel_j);
        end
    end 
end

function surrogate_pli = weighted_phase_lag_index_surrogate(data)
    % Given a multivariate data, returns phase lag index matrix
    % Modified the mfile of 'phase synchronization'
    ch=size(data,2); % column should be channel
    splice = randi(length(data));  % determines random place in signal where it will be spliced

    a_sig=hilbert(data);
    a_sig2= [a_sig(splice:length(a_sig),:); a_sig(1:splice-1,:)];  % %This is the randomized signal
    surrogate_pli=ones(ch,ch);

    for c1=1:ch-1
        for c2=c1+1:ch
            c_sig=a_sig(:,c1).*conj(a_sig2(:,c2));

            numer=abs(mean(imag(c_sig))); % average of imaginary
            denom=mean(abs(imag(c_sig))); % average of abs of imaginary

            surrogate_pli(c1,c2)=numer/denom;
            surrogate_pli(c2,c1)=surrogate_pli(c1,c2);
        end
    end 
end