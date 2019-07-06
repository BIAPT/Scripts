%% Step 1: Do pca on all the features and collect the pcs and the variance explained
data = load('data.mat');
X = data.X;
[X_pca,coeff,var_explained] = get_data_pca(X,80);
%% Step 2: Calculate each pc using each rows and transform the dataxfeature into a dataxpcs

%% Step 3: Train the models using the dataxpcs and check how better the classification error is

%% Step 4: Compare both PLI and AEC and how many pcs each of them have

%% Step 5: concatenate both AEC and PLI into one data frame using the minimum window for each participant

%% Step 6: Repeat the above with the concatenated features sets.


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
        disp(i)
        for j = 1:number_data_points
            X_pca(j,i) = X(j,:) * filtered_pc(i,:)';
        end
    end
    
    
end