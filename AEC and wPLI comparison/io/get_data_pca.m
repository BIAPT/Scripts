function [X_pca,coeff,var_explained] = get_data_pca(X,max_var)
    % Variable initialization
    sum_variance = 0;
    index = 1;
    filtered_pc = [];

    % Calculate the PCA
    [coeff,~,~,~,var_explained,~] = pca(X);
    
    % Keep only the pcs explaining the max_var
    while(sum_variance < max_var && sum_variance < 99.99)
        filtered_pc = [filtered_pc; coeff(index,:)]; 
        sum_variance = sum_variance + var_explained(index);
        index = index + 1;
    end
    
    % Making new dataXpc using the original data
    number_pc = size(filtered_pc,1);
    number_data_points = size(X,1);
    X_pca = zeros(number_data_points,number_pc);
    
    for i = 1:number_pc
        for j = 1:number_data_points
            X_pca(j,i) = X(j,1:length(filtered_pc)) * filtered_pc(i,:)';
        end
    end
    
    
end