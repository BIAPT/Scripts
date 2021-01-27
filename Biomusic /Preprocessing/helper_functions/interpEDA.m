function [EDA_interp] = interpEDA(EDA_data)
% This function interpolates EDA data based on procedure from Kleckner et al.:
% Rules for determining invalid data:
%   1. EDA is out of valid range (e.g., not within 0.05-60 uS)
%   2. EDA changes too quickly (e.g., faster than +-10 ?S/sec)
%   3. EDA data surrounding (e.g., within 4 sec of) invalid portions from rules 1-2 are also invalid
%
% This function is called by preprocessEDA.m
%
% Input: EDA_data - single column EDA data points
% Output: EDA_interp - interpolated EDA 
%
% Dannie Fu September 24 2020
% ----------------------

% Find outliers (Find outliers 3 std away from mean over a 1 second moving window)
TF = isoutlier(EDA_data, 'movmean',15);
outliers = find(TF); % Index of outliers 

% Calculate instantaneous slope 
slope = [0; diff(EDA_data) ./ 15];

% Find idx of all points that are out of range and where the slope changes too
% rapidly
EDA_invalid = find(EDA_data < 0.05 | EDA_data > 60 | abs(slope) > 10);
EDA_invalid = [EDA_invalid; outliers];

% Data within 4 seconds (60 samples) of invalid EDA is also set to NaN.
radius = 30;
for d = 1:length(EDA_invalid)
    
    if ( EDA_invalid(d) - radius < 1 ) 
        idx_bad = 1:EDA_invalid(d) + radius;
    elseif ( EDA_invalid(d) + radius > length(EDA_data) ) 
        idx_bad = EDA_invalid(d) - radius:EDA_invalid(end);
    else
        idx_bad = EDA_invalid(d) - radius:EDA_invalid(d) + radius;
    end 
    
    EDA_data(idx_bad) = NaN;
end

% If EDA is NaN shorter than 30 seconds (450 samples), interpolate with cubic spline
EDA_interp = interp1gap(EDA_data,450,'pchip');

% If EDA is NaN longer than 30 seconds, fill with previous non NaN value. 
EDA_interp = fillmissing(EDA_interp,'previous');

end 