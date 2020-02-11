function [filtered_data] = exponential_decay(data, alpha)
    %Exponential decay smoothing filter
    %increase alpha for more filtering: uses more past data to compute an
    %average
    filtered_data = data;
    for i = 1:(length(data)-1)
        filtered_data(i+1) = (filtered_data(i)*alpha) + (data(i+1)*(1-alpha));
    end 
end

