function [result] = na_phase_amplitude_coupling(recording, window_size, step_size, low_frequency_bandwidth, high_frequency_bandwidth, number_bins)
    %NA_PHASE_AMPLITUDE_COUPLING NeuroAlgo implementation of spr that works with Recording
    % NOTE: right now we are only doing non-overlapping window (in sec)
    
    %% Getting configuration
    configuration = get_configuration();
    
    %% Setting Result
    result = Result('phase amplitude coupling', recording);
    result.parameters.window_size = window_size;
    result.parameters.low_frequency_bandwidth = low_frequency_bandwidth;
    result.parameters.high_frequency_bandwidth = high_frequency_bandwidth;
    result.parameters.number_bins = number_bins;
    
    %% Variable Initialization
    sampling_rate = recording.sampling_rate;
    channels_location = recording.channels_location;
     
    % Here we init the sliding window slicing 
    recording = recording.init_sliding_window(window_size, step_size);
    number_window = recording.max_number_window;
    
   if(~isempty(recording.channels_location))
        anterior_mask = ([channels_location.is_anterior] == 1);
        posterior_mask = ([channels_location.is_posterior] == 1);
   end
    
    %% Calculation on the windowed segments
    result.data.modulogram_all = zeros(number_window, number_bins);
    result.data.ratio_peak_through_all = zeros(1, number_window);
    result.data.modulogram_anterior = zeros(number_window, number_bins);
    result.data.ratio_peak_through_anterior = zeros(1, number_window);
    result.data.modulogram_posterior = zeros(number_window, number_bins);
    result.data.ratio_peak_through_posterior = zeros(1,number_window);
    
    for i = 1:number_window
        print_message(strcat("Phase Amplitude Coupling at window: ",string(i)," of ", string(number_window)),configuration.is_verbose); 
        [recording, segment_data] = recording.get_next_window();
       
        % Whole head
        [modulogram_all, ratio_peak_through_all] = phase_amplitude_coupling(segment_data,sampling_rate, low_frequency_bandwidth, high_frequency_bandwidth, number_bins);
        result.data.modulogram_all(i,:) = modulogram_all;
        result.data.ratio_peak_through_all(i) = ratio_peak_through_all;
        
        % if we don't have channels location we skip the part below
        if(isempty(recording.channels_location))
            continue
        end
        % Only the anterior part
        anterior_segment_data = segment_data(anterior_mask,:);
        [modulogram_anterior, ratio_peak_through_anterior] = phase_amplitude_coupling(anterior_segment_data,sampling_rate, low_frequency_bandwidth, high_frequency_bandwidth, number_bins);
        result.data.modulogram_anterior(i,:) = modulogram_anterior;
        result.data.ratio_peak_through_anterior(i) = ratio_peak_through_anterior;
        
        % Only the posterior part
        posterior_segment_data = segment_data(posterior_mask,:);
        [modulogram_posterior, ratio_peak_through_posterior] = phase_amplitude_coupling(posterior_segment_data,sampling_rate, low_frequency_bandwidth, high_frequency_bandwidth, number_bins);
        result.data.modulogram_posterior(i,:) = modulogram_posterior;
        result.data.ratio_peak_through_posterior(i) = ratio_peak_through_posterior;
    end
    
end