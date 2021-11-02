function [modulogram, ratio_peak_through] = phase_amplitude_coupling(eeg_data,sampling_rate, low_frequency_bandwith, high_frequency_bandwith, number_bins)
%PHASE_AMPLITUDE_COUPLING calculate the PAC on the eeg data and return the
%ratio peak through for the selected frontal and parietal electrodes
%   Input:
%       eeg_data: data to calculate the measures on
%       sampling_rate: sampling frequency at which the data was gathered
%       low_frequency_bandwith: array of size two with the bandwith for the
%       extra low bandwith.
%       high_frequency_bandwith: array of size two with the bandwith for
%       the high frequency bandwith
%       number_bins: number of bins for the modulogram (18 is good)
%   Output:
%       modulogram: the modulogram of the phase and amplitude coupling
%       ratio_peak_through: peak / through;
    
%% Calculate Modulogram
    number_channels = size(eeg_data,1);
    [lfo_phase,hfo_amplitude] = extract_lfo_hfo(eeg_data,sampling_rate,low_frequency_bandwith,high_frequency_bandwith);
    [modulogram] = calculate_modulogram(number_bins,lfo_phase,hfo_amplitude,number_channels);

%% Find the through and peak
    % calculate start index and stop index for peak and through
    start_index_peak = floor(number_bins/4);
    stop_index_peak = floor(3*number_bins/4);
    start_index_through = [1,stop_index_peak+1];
    stop_index_through = [start_index_peak-1,number_bins];
    % Get the peak and through out of the modulogram for frontal and
    % parietal channels and average them
    peak = mean(modulogram(start_index_peak:stop_index_peak));
    through = mean([modulogram(start_index_through(1):stop_index_through(1)); modulogram(start_index_through(2):stop_index_through(2))]);
    
    ratio_peak_through = peak/through;
end

function [low_frequency_phase,high_frequency_amplitude] = extract_lfo_hfo(eeg_data,sampling_rate,low_frequency_bandwith,high_frequency_bandwith)
    %LFO filtering
    eeg_low_frequency = filter_bandpass(eeg_data',sampling_rate,low_frequency_bandwith(1),low_frequency_bandwith(2));
    eeg_low_frequency = eeg_low_frequency';
    
    %HFO filtering
    eeg_high_frequency = filter_bandpass(eeg_data',sampling_rate, high_frequency_bandwith(1),high_frequency_bandwith(2));
    eeg_high_frequency = eeg_high_frequency';
    
    % Calculate the LFO phase and HFO amplitude 
    low_frequency_phase = angle(hilbert(eeg_low_frequency)); %Take the angle of the Hilbert to get the phase
    high_frequency_amplitude = abs(hilbert(eeg_high_frequency)); %calculating the amplitude by taking absolute value of hilber

end

function [modulogram] = calculate_modulogram(number_bins,low_frequency_phase,...,
                                            high_frequency_amplitude,number_channels)
    %% Variable Setup
    bin_size = (2*pi)/number_bins; 
    % ! Speedup can be gained over here, leave that for now  ! %
    sorted_amplitude = zeros(number_bins,2);
   
    %% Bin Sorting
    % Here we sort amplitude according to the phase
    for channel = 1:number_channels
        channel_phase = low_frequency_phase(channel,:);
        channel_amplitude = high_frequency_amplitude(channel,:);
        
        for point_index = 1:length(channel_phase)
            % Incrementing the amplitude and putting in the right bin
            for bin = 1:number_bins
                if(is_phase_in_range(channel_phase,point_index,bin,bin_size))
                    sorted_amplitude(bin,1) = sorted_amplitude(bin) + channel_amplitude(point_index);
                    sorted_amplitude(bin,2) = sorted_amplitude(bin,2) + 1;
                end
            end
        end
    end
    
    %% Averaging
    % Calculate the average amplitude
    avg_sorted_amplitude = zeros(number_bins,1);
    for bin = 1:number_bins
            if sorted_amplitude(bin,2) == 0
                avg_sorted_amplitude(bin) = 0;
            else
                avg_sorted_amplitude(bin) = (sorted_amplitude(bin,1)/sorted_amplitude(bin,2));
            end
    end   
    avg_amplitude = mean(avg_sorted_amplitude);
    
    %% Create Modulogram
    % For each bins set the value at that position
    modulogram = zeros(number_bins,1);
    for bin = 1:number_bins
        modulogram(bin) = ((avg_sorted_amplitude(bin)-avg_amplitude)/avg_amplitude) + 1;
    end   
    
    %% Filtering
    % Filter the modulogram
    modulogram = modulogram - 1;            % Do this because median filter assumes 0 on each side
    modulogram = medfilt1(modulogram, 2);   % January 16, 2014
    modulogram = modulogram + 1;
end

function [is_in_range] = is_phase_in_range(channel_phase,point_index,bin,bin_size)
    is_in_range = channel_phase(point_index) > -pi + (bin-1)*bin_size && channel_phase(point_index) < -pi + (bin)*bin_size;
end

