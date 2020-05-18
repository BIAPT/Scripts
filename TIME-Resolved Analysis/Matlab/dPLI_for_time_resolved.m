%% Loading a .set file
% This will allow to load a .set file into a format that is amenable to analysis
% The first argument is the name of the .set you want to load and the
% second argument is the path of the folder containing that .set file
% Here I'm getting it programmatically because my path and your path will
% be different.
filepath = '/Users/biapt/Desktop/Time_Resolved_EEG/data/';
fileList = dir('*.set');

data_path = '/Users/biapt/Desktop/Time_Resolved_EEG/dPLI_10_1/';

for p=1:length(fileList)
    name=fileList(p).name(1:length(fileList(p).name)-4);
    recording = load_set(char(fileList(p).name),'');
    disp(string(fileList(p).name)+" load complete ========================================" )
    
    % wPLI
    frequency_band = [7 13]; % This is in Hz
    window_size = 10; % This is in seconds and will be how we chunk the whole dataset
    number_surrogate = 10; % Number of surrogate wPLI to create
    p_value = 0.05; % the p value to make our test on
    step_size = 1;
    
    result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
    result_dpli = result_dpli.data.dpli;
    save(data_path+"/"+name+"dPLI_10_1",'result_dpli')
    
end

    
% wPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = 1;

result_wpli_09base = na_wpli(recording_09base, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_09anes = na_wpli(recording_09anes, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_09reco = na_wpli(recording_09reco, frequency_band, window_size, step_size, number_surrogate, p_value);

result_wpli_10base = na_wpli(recording_10base, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_10anes = na_wpli(recording_10anes, frequency_band, window_size, step_size, number_surrogate, p_value);
result_wpli_10reco = na_wpli(recording_10reco, frequency_band, window_size, step_size, number_surrogate, p_value);

result_wpli_10anes_step=result_wpli_10anes.data.wpli;
result_wpli_10base_step=result_wpli_10base.data.wpli;
result_wpli_10reco_step=result_wpli_10reco.data.wpli;


result_wpli_09anes_step=result_wpli_09anes.data.wpli;
result_wpli_09base_step=result_wpli_09base.data.wpli;
result_wpli_09reco_step=result_wpli_09reco.data.wpli;





% dPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = 1;

result_dpli_13base = na_dpli(recording_13base, frequency_band, window_size, step_size, number_surrogate, p_value);
result_dpli_13anes = na_dpli(recording_13anes, frequency_band, window_size, step_size, number_surrogate, p_value);
result_dpli_13reco = na_dpli(recording_13reco, frequency_band, window_size, step_size, number_surrogate, p_value);
result_dpli_20base = na_dpli(recording_20base, frequency_band, window_size, step_size, number_surrogate, p_value);
result_dpli_20anes = na_dpli(recording_20anes, frequency_band, window_size, step_size, number_surrogate, p_value);
result_dpli_20reco = na_dpli(recording_20reco, frequency_band, window_size, step_size, number_surrogate, p_value);

result_dpli_20anes_step=result_dpli_20anes.data.dpli;
result_dpli_20base_step=result_dpli_20base.data.dpli;
result_dpli_20reco_step=result_dpli_20reco.data.dpli;


result_dpli_13anes_step=result_dpli_13anes.data.dpli;
result_dpli_13base_step=result_dpli_13base.data.dpli;
result_dpli_13reco_step=result_dpli_13reco.data.dpli;
