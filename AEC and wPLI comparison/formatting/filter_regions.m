function [filtered_data] = filter_regions(data,regions_mask)
%FILTER Summary of this function goes here
%   Detailed explanation goes here
    % Filtering the data
    filtered_data = data;
    return;
    number_good_regions = length(regions_mask(regions_mask == 1));
    filtered_data = zeros(size(data,1),number_good_regions,number_good_regions);
    for i = 1:size(data,1)
        region_j = 1;
        for j = 1:length(regions_mask)
           if(regions_mask(j) == 1)
              region_k = 1;
              for k = 1:length(regions_mask)
                 if(regions_mask(k) == 1)
                    filtered_data(i,region_j,region_k) = data(i,j,k);
                    region_k = region_k + 1;
                 end
              end
              region_j = region_j + 1;
           end
       end
    end
end

