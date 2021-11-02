%% Loading a .set file
% This will allow to load a .set file into a format that is amenable to analysis
% The first argument is the name of the .set you want to load and the
% second argument is the path of the folder containing that .set file
% Here I'm getting it programmatically because my path and your path will
% be different.
[filepath,name,ext] = fileparts(mfilename('fullpath'));
test_data_path = strcat(filepath,'/test_data');
recording = load_set('test_data.set',test_data_path);

%recording = load_set('test_data.set',test_data_path);
%{ 
    The recording class is structured as follow:
    recording.data = an (channels, timepoints) matrix corresponding to the EEG
    recording.length_recoding = length in timepoints of recording
    recording.sampling_rate = sampling frequency of the recording
    recording.number_channels = number of channels in the recording
    recording.channels_location = structure containing all the data of the channels (i.e. labels and location in 3d space)
    recording.creation_data = timestamp in UNIX format of when this class was created
%}

%% Running the analysis and Plot connectivity circles
%{
    These functions will plot the wPLI into a connectivity
    circle, the locations of the electrodes on the circle are specified
    using the excel file "electrodes_numbers and regions_new". Missing
    electrodes in individual subjects are adapted accordingly and leave a
    white space in the Circle.  

    Currently available are only wPLI (dPLI will follow) in inter- and
    intra-hemispheric relation or midline-connections

%}

% FIRST: calculate wPLI
% wPLI
frequency_band = [7 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 10; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
%step_size = window_size;
step_size = 10;
result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);


%plot an empty default circle
%plot will contain all connectivity LARGER then this (plot if value > threshold)
%the threshold must be between 0 and maxvalue
threshold=0.15;

mode="Inter"; % Mode can be "Inter" for interhemispheric connections,
                %"Intra" for Intrahemisoheric connections and
                %"Midline" for connections to the Midlien-electrodes

% The excel file provedes the location and electrode ordering in the circle. 
% Make sure the excel file is in the current working directory
% This will generate a single image  
plot_circle(result_wpli.data.wpli(2,:,:),recording,threshold, mode);
plot_circle(result_wpli.data.avg_wpli,recording,threshold, mode);

%This will generate a video of connectivity
%The video will be saved in the current directory; 
% A step size of 0.1 -1s is a good value for a flluent video 
name = 'test_test'
make_video_circle(name, result_wpli.data.wpli(:,:,:),recording,threshold, mode, step_size)


% Note: This code is still in development, dPLI visualization will follow. 