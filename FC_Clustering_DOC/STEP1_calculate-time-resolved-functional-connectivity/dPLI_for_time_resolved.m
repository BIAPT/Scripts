%% Loading a .set file
% This will allow to load a .set file into a format that is amenable to analysis
% The first argument is the name of the .set you want to load and the
% second argument is the path of the folder containing that .set file
% Here I'm getting it programmatically because my path and your path will
% be different.
%filepath = '/Users/BIAPT/Documents/Time-Resolved-dPLI/DOC/BASELINE_5min_250Hz/';
filepath = '/Users/BIAPT/Documents/GitHub/Scripts/FC_Clustering_DOC/data/BASELINE_5min_250Hz/';
fileList = dir(fullfile(filepath, '*.set'));
%fileList = filepath('*.set');

data_path = '/Users/BIAPT/Documents/GitHub/Scripts/FC_Clustering_DOC/data/dPLI_10_10_alpha';

for p=1:length(fileList)
    name = fileList(p).name(1:length(fileList(p).name)-4);
    recording = load_set(fullfile(filepath, char(fileList(p).name)),'');
    disp(string(fileList(p).name)+" load complete ========================================" )
    name = fileList(p).name(1:length(fileList(p).name)-9);
    
    % dPLI
    frequency_band = [8 13]; % This is in Hz
    window_size = 10; % This is in seconds and will be how we chunk the whole dataset
    number_surrogate = 10; % Number of surrogate dPLI to create
    p_value = 0.05; % the p value to make our test on
    step_size = 10;
    
    result_dPLI = na_dPLI(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
    result_dPLI = result_dPLI.data.dPLI;
    save(data_path+"/"+name+"_dPLI_10_1_alpha",'result_dPLI')
    
end

