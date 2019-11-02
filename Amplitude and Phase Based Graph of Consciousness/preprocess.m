%{
    Yacine Mahdid 2019-11-01
    This is code taken from the AEC vs wPLI scripts repository, stripped
    down for what is really needed and augmented with the graph theory
    features

    -> We will be generating both the AEC and the wPLI
    -> We will also NOT be doing the normalization here (so we can
    experiment with normalization on the machine learning part of the
    project)
    -> We will keep all the target epochs, there is no need to limit
    ourselves here
%}

%% Variable Initialization
%% TODO: KEEP ON CLEANING THIS, you might need to bring stuff from the other sripts

%
% Data file paths
data_path = "/home/yacine/Documents/AEC vs PLI/data";
scout_path = "/home/yacine/Documents/AEC vs PLI/scouts.mat";

% Experiment Parameters
target_frequency = "alpha";
target_type = "both";
target_epoch = {'ec1','emf5','ec1','if5','emf5','eml5','ec3','ec8'};

%% Data Loading
disp('Load data');

% Scouts
scout = load(scout_path);
scout = scout.scout;

regions_mask = zeros(1,length(scout));

for i=1:length(scout)
    regions_mask(i) = scout(i).isKeep;
end

% Data (PLI or/and AEC) 
files = dir(data_path);
full_dataset = [];
full_dataset_pli = [];
full_dataset_aec = [];
full_label = [];
for file = files'
    if(~strcmp(file.name,'.') && ~strcmp(file.name,'..'))
        data_path = strcat(file.folder,filesep,file.name);
        
        [type,epoch,frequency] = get_content(file.name);
        if(strcmp(target_type,"both") && strcmp(type,"pli") && strcmp(frequency,target_frequency) && any(strcmp(target_epoch,epoch)))
           [data,data_pli,data_aec,label,current_identity] = get_both_data(epoch,frequency,regions_mask,file.folder); 
           
           if(isempty(full_dataset))
              full_dataset = data; 
              full_dataset_pli = data_pli;
              full_dataset_aec = data_aec;
              full_label = label;
              identity = current_identity;
           else
              full_dataset = [full_dataset;data];
              full_dataset_pli = [full_dataset_pli;data_pli];
              full_dataset_aec = [full_dataset_aec;data_aec];
              full_label = [full_label,label];
              identity = [identity,current_identity];
           end
        elseif(strcmp(type,target_type) && strcmp(frequency,target_frequency) && any(strcmp(target_epoch,epoch)))
           data = load(data_path);
           [data,label,current_identity] = get_data(data,type,regions_mask,epoch);

           if(isempty(full_dataset))
              full_dataset = data; 
              full_label = label;
              identity = current_identity;
           else
              full_dataset = [full_dataset;data];
              full_label = [full_label,label];
              identity = [identity,current_identity];
           end
       end
       
    end
end

%% Classification (Training and Validation)
% Data preparation
X = full_dataset;
Y = full_label;
I = identity;

disp('Saving Data');
if(strcmp(target_type,"both"))
    % Save the data for further processing in Python
    save('data/Y','Y');
    save('data/I','I');
    
    X = full_dataset_pli;
    X = (X - min(X)) ./ (max(X) - min(X));
    target_type = 'pli';
    save(strcat('data/X_',target_type),'X');
    
    X = full_dataset_aec;
    X = (X - min(X)) ./ (max(X) - min(X));
    target_type = 'aec';
    save(strcat('data/X_',target_type),'X'); 
else  
    % Save the data for further processing in Python
    save(strcat('data/X_',target_type),'X');
    save(strcat('data/Y_',target_type),'Y');
    save(strcat('data/I_',target_type),'I');   
end


disp('Done!')