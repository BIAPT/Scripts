function [avg_data] = exp_decay(data,alpha)
%Exponential decay smoothing filter
%increase alpha for more filtering: uses more past data to compute an
%average
avg_data = data;
for i = 1:(length(data)-1)
    data(i+1)=(data(i)*alpha) + (data(i+1)*(1-alpha));
    avg_data(i,1)=data(i+1);
end 