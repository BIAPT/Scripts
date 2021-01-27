function [unpadded_data,unpadded_time] = unpad(data,time,timeStart,timeEnd)
% Takes a two column matrix with time in the first column and the signal in
% the second column and removes a chunk in the beginning before the
% specified start time

[~, idxStart] = min(abs(time - timeStart));
[~, idxEnd] = min(abs(time - timeEnd));


unpadded_data = data(idxStart:idxEnd);
unpadded_time = time(idxStart:idxEnd);

end