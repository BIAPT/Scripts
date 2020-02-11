function [avg_spectrum, spectrum, timestamp, frequency, peak_frequency] = spectral_power(eeg_data,sampling_rate, frequency_band,time_bandwith_product,number_tapers,window_size,step_size)
%SPECTRAL_POWER_RATIO calculate the spectral power ratio between the beta
%and alpha band & between the alpha and theta band
    
    %% Setup Variables
    eeg_data = eeg_data';
    %% Create params struct for Chronux function
    params.tapers = [time_bandwith_product number_tapers];
    params.Fs = sampling_rate;
    params.trialave = 1;
    window_parameters = [window_size step_size];    

    %% Spectral Power
    params.fpass = frequency_band;
    [spectrum, timestamp, frequency] = mtspecgramc(eeg_data, window_parameters, params);
    overall_spectrum = mean(spectrum,2);
    avg_spectrum = mean(overall_spectrum);
    
    %% Peak Frequency
    [~,frequency_id] = max(spectrum, [], 2);
    peak_frequency = frequency(frequency_id)';
end

