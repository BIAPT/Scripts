function I = morans_I(grid,weight_matrix)

    if (size(grid,1) > 1)
        x = reshape(grid,[1 size(weight_matrix,1)]);
    else
        x = grid;
    end
    
    ind = find(~isnan(x)); %indices of non-NaN elements in grid
    N = size(ind,2);
    W = 0;
    
    Xbar = nanmean(x);
    num = 0;
    denom = 0;
    for i = 1:size(weight_matrix,1)
        for j = 1:size(weight_matrix,2)
            num = nansum([num weight_matrix(i,j)*(x(i)-Xbar)*(x(j)-Xbar)]);
            if ~isnan(x(i)) && ~isnan(x(j))
                W = W + weight_matrix(i,j);
            end
        end
        denom = nansum([denom (x(i)-Xbar)^2]);
    end
    I = (N/W)*(num/denom);
end
