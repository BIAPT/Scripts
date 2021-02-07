function [HR_avefilt, HR_cubic, HRV_YZ_avefilt, HRV_YZ_interp, HRV_YZ_cubic, HRV_X_avefilt, HRV_X_interp, HRV_X_cubic, HRV_Y_avefilt, HRV_Y_interp, HRV_Y_cubic, HRV_Z_avefilt, HRV_Z_interp, HRV_Z_cubic] = preprocessHR(HR_data, HR_time, HRV_YZ, HRV_X, HRV_Y, HRV_Z, HRV_time)
% This function preprocess HR and HRV data:
% Moving average filter 
% Cubic spline filter 
%
% Inputs: 
%   HR_data - single column HR data
%   HR_time - single column time points for HR data
%   HRV_YZ - single column HRVZY data
%   HRV_X - single column HRVX data
%   HRV_Y - single column HRVY data
%   HRV_Z - single column HRVZ data
%   HRV_time - single column time points for HRV data
% 
% Dannie Fu August 5 2020
% ---------------------

% Moving average filter 
HR_avefilt = movmean(HR_data,5);
HRV_X_avefilt = movmean(HRV_X,5);
HRV_Y_avefilt = movmean(HRV_Y,5);
HRV_Z_avefilt = movmean(HRV_Z,5);
HRV_YZ_avefilt = movmean(HRV_YZ,5);

% If HRV is NaN shorter than 30 seconds interpolate with cubic spline 
HRV_X_interp = interp1gap(HRV_X_avefilt,450,'pchip');
HRV_Y_interp = interp1gap(HRV_Y_avefilt,450,'pchip');
HRV_Z_interp = interp1gap(HRV_Z_avefilt,450,'pchip');
HRV_YZ_interp = interp1gap(HRV_YZ_avefilt,450,'pchip');

% IF HRV is NaN longer than 30 seconds, interpolate with previous non NaN value value
HRV_X_interp = fillmissing(HRV_X_interp,'previous');
HRV_Y_interp = fillmissing(HRV_Y_interp,'previous');
HRV_Z_interp = fillmissing(HRV_Z_interp,'previous');
HRV_YZ_interp = fillmissing(HRV_YZ_interp,'previous');

% Cubic splining function 
HR_cubic = csaps(HR_time,HR_avefilt',0.001,HR_time); 
HRV_X_cubic = csaps(HRV_time,HRV_X_interp',0.001,HRV_time);
HRV_Y_cubic = csaps(HRV_time,HRV_Y_interp',0.001,HRV_time);
HRV_Z_cubic = csaps(HRV_time,HRV_Z_interp',0.001,HRV_time);
HRV_YZ_cubic = csaps(HRV_time,HRV_YZ_interp',0.001,HRV_time);

end 