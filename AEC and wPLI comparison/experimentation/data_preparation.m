% Variable Initialization
data_path = "C:\Users\Fabien Dal Maso\Documents\Machine Learning Yacine 2018-2019\AEC vs PLI\data";
scout_path = "C:\Users\Fabien Dal Maso\Documents\Machine Learning Yacine 2018-2019\AEC vs PLI\scouts.mat";
target_frequency = "alpha";
target_type = "both";
is_pca = 0;
pca_threshold = 80;

% Loading required data
scout = load(scout_path);
scout = scout.scout;
regions_mask = zeros(1,length(scout));
for i=1:length(scout)
    regions_mask(i) = scout(i).isKeep;
end

% loading data in data path 
files = dir(data_path);
full_dataset = [];
full_label = [];
for file = files'
    if(~strcmp(file.name,'.') && ~strcmp(file.name,'..'))
        file_name = file.name;
        data_path = strcat(file.folder,filesep,file_name);
        
        [type,epoch,frequency] = get_content(file_name);
        if(strcmp(type,target_type) && strcmp(frequency,target_frequency))
           disp(file_name);
           %[data,label] = get_both_data(epoch,frequency,regions_mask,file.folder);
           data = load(data_path);
           [data,label] = get_data(data,type,regions_mask,epoch);

           if(isempty(full_dataset))
              full_dataset = data; 
              full_label = label;
           else
              full_dataset = [full_dataset;data];
              full_label = [full_label,label];
           end
       end
       
    end
end

X = full_dataset;
Y = full_label;

if(is_pca)
    [X,coeff,var_explained] = get_data_pca(X,pca_threshold);
end

% Train an error-correcting output codes (ECOC) multiclass model using support vector machine (SVM) binary learners.
% we use 5 fold cross validation for parameters optimization and 5 fold
% cross validation for the calculation of the error (we try to avoid
% overfit as much as possible)
disp('Training Linear SVM');
linear_svm = templateSVM('Standardize',1);
model_linear = fitcecoc(X,Y,'Learners',linear_svm,'FitPosterior',1,'Verbose',2);
disp('Training Gaussian SVM');
gaussian_svm = templateSVM('Standardize',1,'KernelFunction','gaussian');
model_gaussian = fitcecoc(X,Y,'Learners',gaussian_svm,'FitPosterior',1,'Verbose',2);
disp('Cross validating linear model');
cross_validated_linear_model = crossval(model_linear);
loss_model_linear = kfoldLoss(cross_validated_linear_model)
disp('Cross Validating Gaussian model');
cross_validated_gaussian_model = crossval(model_gaussian);
loss_model_gaussian = kfoldLoss(cross_validated_gaussian_model)


function [type,epoch,frequency] = get_content(file_name)
        content  = strsplit(file_name,'_');
        type = content{1};
        epoch = content{2};
        
        content = strsplit(content{3},'.');
        frequency = content{1};
end

function [data,label] = get_both_data(epoch,frequency,regions_mask,folder)
    aec_file_name = strcat(folder,filesep,"aec_",epoch,"_",frequency,".mat");
    pli_file_name = strcat(folder,filesep,"pli_",epoch,"_",frequency,".mat");
    aec_data = load(aec_file_name);
    pli_data = load(pli_file_name);
    
    [pli_data,pli_label] = get_data(pli_data,"pli",regions_mask,epoch);
    [aec_data,aec_label] = get_data(aec_data,"aec",regions_mask,epoch);
    data = cat(2,aec_data,pli_data);
    label = cat(2,aec_label,pli_label);
end

function [flatten_data,label] = get_data(data,type,regions_mask,epoch)
    % Get the data
    if(strcmp(type,"pli"))
       data = data.PLI_OUT;
        % Getting the data out of the cells and concatenating them together
        merged_data = data{1};
        for i = 2:length(data)
            merged_data = cat(1,merged_data,data{i});
        end
    elseif(strcmp(type,"aec"))
        data = data.AEC_OUT;
        merged_data = data{1};
        for i = 2:length(data)
            merged_data = cat(3,merged_data,data{i});
        end
        merged_data = permute(merged_data,[3 2 1]);      
    end
    
    % Filter out the regions we are not using
    number_good_regions = length(regions_mask(regions_mask == 1));
    filtered_data = zeros(length(merged_data),number_good_regions,number_good_regions);
 
    % Filtering the data
    for i = 1:length(merged_data)
        region_j = 1;
        for j = 1:length(regions_mask)
           if(regions_mask(j) == 1)
              region_k = 1;
              for k = 1:length(regions_mask)
                 if(regions_mask(k) == 1)
                    filtered_data(i,region_j,region_k) = merged_data(i,j,k);
                    region_k = region_k + 1;
                 end
              end
              region_j = region_j + 1;
           end
       end
    end
   
    % Flatten the 3d array so the data is in a 2d array FeatureXdata points
    flatten_data = reshape(filtered_data,[],size(filtered_data,1),1)';
    
    % Make the label
    if(strcmp(epoch,"ec1"))
        number_label = 0;
    elseif(strcmp(epoch,"if5"))
        number_label = 1;
    elseif(strcmp(epoch,"emf5"))
        number_label = 2;
    elseif(strcmp(epoch,"eml5"))
        number_label = 3;
    elseif(strcmp(epoch,"ec3"))
        number_label = 0;
    end
    label = repelem(number_label,length(merged_data));
end

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