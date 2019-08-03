function C = gearys_C(grid,weight_matrix)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes heres

    if (size(grid,1) > 1)
        x = reshape(grid,[1 size(weight_matrix,1)]);
    else
        x = grid;
    end
    
    
    % TODO change these two to not take the NAN into consideration
    N = size(weight_matrix,1);
    W = sum(sum(weight_matrix));
    
    Xbar = nanmean(x);
    num = 0;
    denom = 0;
    for i = 1:size(weight_matrix,1)
        for j = 1:size(weight_matrix,2)
            num = nansum([num weight_matrix(i,j)*(x(i)-x(j))]);
        end
        denom = nansum([denom (x(i)-Xbar)^2]);
    end
    C = ((N-1)*num)/((2*W)*denom);
end

