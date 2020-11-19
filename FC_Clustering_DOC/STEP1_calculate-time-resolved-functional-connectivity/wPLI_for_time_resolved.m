% Charlotte Maschke 11.11.2020
%% Generate time-resolved weighted Phase Lag Index
% this code is optimized to be run on multiple cores, for example on ComputeCanada

%% Please adapt the following paths to your location: 
INPUT_DIR = "/home/lotte/projects/def-sblain/lotte/DOC_cluster/data/BASELINE_5min_250Hz/";
OUTPUT_DIR = "/home/lotte/projects/def-sblain/lotte/DOC_cluster/results/graphs/";
NEUROALGO_PATH = "/home/lotte/projects/def-sblain/lotte/DOC_cluster/NeuroAlgo";
addpath(genpath(NEUROALGO_PATH)); % Add NA library to our path so that we can use it

% This list contains all participant IDs
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17',...
    'WSAS02', 'WSAS05', 'WSAS07', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13','WSAS15','WSAS16','WSAS17',...
    'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22','WSAS23',...
    'AOMW03','AOMW04','AOMW08','AOMW22','AOMW28','AOMW31','AOMW34','AOMW36'};

filepath = '/Users/BIAPT/Documents/GitHub/Scripts/FC_Clustering_DOC/data/BASELINE_5min_250Hz/';
fileList = dir(fullfile(filepath, '*.set'));
data_path = '/Users/BIAPT/Documents/GitHub/Scripts/FC_Clustering_DOC/data/wPLI_10_10_alpha';


%for s = 1:len(step_size)

parfor p=1:length(fileList)
    name = fileList(p).name(1:length(fileList(p).name)-4);
    recording = load_set(fullfile(filepath, char(fileList(p).name)),'');
    disp(string(fileList(p).name)+" load complete ========================================" )
    name = fileList(p).name(1:length(fileList(p).name)-9);
    
    % wPLI
    frequency_band = [8 13]; % This is in Hz
    window_size = 10; % This is in seconds and will be how we chunk the whole dataset
    number_surrogate = 10; % Number of surrogate dPLI to create
    p_value = 0.05; % the p value to make our test on
    step_size = 10;
    
    result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
    result_wpli = result_wpli.data.wpli;
    save(data_path+"/"+name+"_wPLI_10_1_alpha",'result_wpli')
    
end


    
