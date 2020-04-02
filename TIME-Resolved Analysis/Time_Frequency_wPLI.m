%% Loading a .set file
%fileList = dir('*.set');

fileList = dir('WSAS10_Base_300.set');

data_path = strcat('','wPLI_30_10');

for p=1:length(fileList)
    name=fileList(p).name(1:length(fileList(p).name)-4);
    recording = load_set(char(fileList(p).name),'');
    disp(string(fileList(p).name)+" load complete ========================================" )

    info=cell(1,5);
    data_step=cell(1,70);
    data_avg=cell(1,70);
    data=cell(1,3);

    bands=0.25:0.5:35.25; % from 0.5 to 35 hz in ranges of 0.5
    window_size = 30; % This is in seconds and will be how we chunk the whole dataset
    number_surrogate = 10; % Number of surrogate wPLI to create
    p_value = 0.05; % the p value to make our test on
    step_size = 10;


    info{1}=bands;
    info{2}=window_size;
    info{3}=number_surrogate;
    info{4}=p_value;
    info{5}=step_size;

    for i=1:length(bands)-1
        lb=bands(i);
        ub=bands(i+1);
        frequency_band = [lb ub]; % This is in Hz
        result_wpli= na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
        data_step{i}=result_wpli.data.wpli;
        data_avg{i}=result_wpli.data.avg_wpli;    
        disp("completed frequency "+string(i)+" of 70")
    end

    data{1}=data_step;
    data{2}=data_avg;
    data{3}=info;

    save(data_path+"/"+name+"wPLI_30_30",'data')
    
end