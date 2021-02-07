function [TEMP_medfilt, TEMP_avefilt, TEMP_interp, TEMP_expfilt] = preprocessTEMP(TEMP_data)
% This script preprocess TEMP data:
% 1D median filter 
% Moving average filter 
% Exponential decay filter 
%
% Input: TEMP_data - single column of TEMP data (no time column)
% Outputs:  
%    TEMP_medfilt - median filtered 
%    TEMP_avefilt - median + ave filtered
%    TEMP_interp - median + ave + interpolated
%    TEMP_expfilt - median + ave + interpolated + exponential filter
%
% Dannie Fu August 5 2020
% ---------------------

% Median filter
TEMP_medfilt = medfilt1(TEMP_data,1,'truncate');

% Moving average 
TEMP_avefilt = movmean(TEMP_medfilt,15);

% If TEMP is NaN shorter than 30 seconds (450 samples), interpolate with cubic spline  
TEMP_interp = interp1gap(TEMP_avefilt,450,'pchip');

% If TEMP is NaN longer than 30 seconds, fill with previous nonNan value 
TEMP_interp = fillmissing(TEMP_interp,'previous');

% Exponential decay filter
TEMP_expfilt = exp_decay(TEMP_interp',0.95);

end 

%%
% plot((TEMP_time-TEMP_time(1))/1000,TEMP_data,'LineWidth',1)
% hold on
% plot((TEMP_time-TEMP_time(1))/1000,TEMP_medfilt,'LineWidth',1)
% hold on 
% plot((TEMP_time-TEMP_time(1))/1000,TEMP_avefilt,'LineWidth',1)
% hold on
% plot((TEMP_time-TEMP_time(1))/1000,TEMP_expfilt,'LineWidth',1)
% ylabel("Temperature (C)")
% legend("raw", "medfilt", "medfilt+avefilt", "expfilt")
% 
