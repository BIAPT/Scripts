function [unpadded_data] = unpadBeginning(data,length,fs)
% Inputs: 
% data - [time, signal]
% length - length to cut in beginning (seconds)
% fs - sampling rate 

duration = length*fs;

unpadded_data(:,1) = data(1+duration:end,1);
unpadded_data(:,2) = data(1+duration:end,2);


end