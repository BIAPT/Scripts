%% Loading a .set file
% This will allow to load a .set file into a format that is amenable to analysis
% The first argument is the name of the .set you want to load and the
% second argument is the path of the folder containing that .set file
% Here I'm getting it programmatically because my path and your path will
% be different.
[filepath,name,ext] = fileparts(mfilename('fullpath'));
test_data_path = strcat(filepath,'/test_data');
recording = load_set('test_data.set',test_data_path);

% dPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

figure;
subplot(2,1,1)
imagesc(squeeze(mean(result_dpli.data.dpli)));
colorbar
colormap jet; 
title(strcat("Average Participant at ", string(number_surrogate), " surrogates"));
subplot(2,1,2);
imagesc(squeeze(result_dpli.data.dpli(15,:,:)));
colormap jet; 
colorbar;
title(strcat("Single participant #",string(15)," at ", string(number_surrogate), " surrogates"));
% wPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);

figure;
subplot(2,1,1)
imagesc(squeeze(mean(result_wpli.data.wpli)));
colorbar
colormap jet; 
title(strcat("Average Participant at ", string(number_surrogate), " surrogates"));
subplot(2,1,2);
imagesc(squeeze(result_wpli.data.wpli(15,:,:)));
colormap jet; 
colorbar;
title(strcat("Single participant #",string(15)," at ", string(number_surrogate), " surrogates"));
