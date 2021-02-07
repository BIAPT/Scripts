function [unpadded_data] = unpadEnd(data,length,fs)
% Takes a two column matrix with time in the first column and the signal in
% the second column and removes a portion at the end of the signal specified by length in seconds 

unpadded_data(:,1) = data(1:end-length*fs,1);
unpadded_data(:,2) = data(1:end-length*fs,2);

end