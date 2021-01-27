function [EDA_medfilt, EDA_avefilt, EDA_interp, EDA_eurofilt] = preprocessEDA(EDA_data)
% This function preprocesses EDA:
%       - 1D median filter 
%       - Moving average filter 
%       - Cubic Spline Interpolation 
%       - 1euro filter 
%
% Input: EDA_data - single column of EDA data (no time column)
% Outputs:  
%    EDA_medfilt - median filtered 
%    EDA_avefilt - median + ave filtered
%    EDA_interp - median + ave + interpolated
%    EDA_eurofilt - median + ave + interpolated + 1euro filter
%
% Dannie Fu August 4 2020
% -----------------------

% 1D median filter
EDA_medfilt = medfilt1(EDA_data,75,'truncate'); 

% Moving average filter 
EDA_avefilt = movmean(EDA_medfilt,10); %Window size 10, arbitrarily chosen 

EDA_interp = interpEDA(EDA_avefilt);

% Apply 1 euro filter 
a = oneEuro; % Declare oneEuro object
a.mincutoff = 50.0; % Decrease this to get rid of slow speed jitter
a.beta = 30.0; % Increase this to get rid of high speed lag

EDA_eurofilt = zeros(size(EDA_interp'));
for i = 1:length(EDA_interp)
    EDA_eurofilt(i) = a.filter(EDA_interp(i),i);      
end

% transpose so it's a column instead of row when returning value
EDA_interp = EDA_interp';
EDA_avefilt = EDA_avefilt';

end 
 