function [power] = topographic_distribution(eeg_data,sampling_rate)
%TOPOGRAPHIC_DISTRIBUTION Calculate the ratio of front to back power at a
%specific frequency
%   Input:
%       eeg_data: data to calculate the measures on
%       eeg_info: headset information
%       parameters: variables data as inputed by the user
%       frontal_mask: boolean mask for the midline electrodes
%       posterior_mask: boolean mask for the lateral electrode
%   Output:
%       ratio_front_back: ratio of the power between front and back
%       electrodes

    %% Spectral topographic map
    [power,~,~,~,~] = spectopo(eeg_data,length(eeg_data),sampling_rate,'plot','off');
    
end

