function [result] = na_spectral_power(recording, window_size, time_bandwidth_product,number_tapers,spectrum_window_size, bandpass, step_size)
    %NA_SPECTRAL_POWER NeuroAlgo implementation of spr that works with Recording
 
    configuration = get_configuration();
    
    %% Setting Result
    result = Result('sprectral power', recording);
    result.parameters.window_size = window_size;
    result.parameters.time_bandwidth_product = time_bandwidth_product;
    result.parameters.number_tapers = number_tapers;
    result.parameters.spectrum_window_size = spectrum_window_size;
    result.parameters.step_size = step_size;
    result.parameters.bandpass = bandpass;
    
    %% Variable Initialization
    sampling_rate = recording.sampling_rate;
    
    % Here we init the sliding window slicing 
    recording = recording.init_sliding_window(window_size, step_size);
    number_window = recording.max_number_window;
    
    %% Calculation on the windowed segments
    result.data.avg_spectrums = zeros(1,number_window);
    result.data.spectrums = [];
    for i = 1:number_window
       print_message(strcat("Spectral Power at window: ",string(i)," of ", string(number_window)),configuration.is_verbose); 
       [recording, segment_data] = recording.get_next_window();
       
       [avg_spectrum,spectrum,timestamp,frequency,peak_frequency] = spectral_power(segment_data,sampling_rate, bandpass, time_bandwidth_product,number_tapers,spectrum_window_size,step_size); 
       
       result.data.avg_spectrums(i) = avg_spectrum; 
       if (i == 1)
          result.data.spectrums = spectrum; 
          result.data.timestamps = timestamp;
          result.data.frequencies = frequency;
          result.data.peak_frequency = peak_frequency;
       else
           result.data.spectrums = cat(3,result.data.spectrums,spectrum);
           result.data.peak_frequency = cat(2,result.data.peak_frequency,peak_frequency);
       end
       %result.data.spectrums = [result.data.spectrums,spectrum]; % TODO: find a way to initialize this properly
    end
    
end
