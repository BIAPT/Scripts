function clean = preprocess(EDA,TEMP,HR,HRV)
% This function preprocesses EDA, TEMP, HR, HRV data. Beginning and ends of
% data are cut if they are 0s. Individual signal specific preprocessing 
% functions are called that filter and smooth the data. 
% All signals are plotted showing each step of the preprocessing.
%
% Input - EDA, TEMP, HR  HRV tables [time data]
% Output - EDA, TEMP, HR, HRV cleaned [time data]
%
% ----------------------------------

% Sometimes the headers in the tables are capitalized or not captilized
try
    EDA_time = EDA.eda_time;
    TEMP_time = TEMP.temp_time;
    HR_time = HR.hr_time;
    HRV_time = HRV.hrv_time;

    EDA_data = EDA.eda_data;
    TEMP_data = TEMP.temp_data;
    HR_data = HR.hr_data;
    HRV_X = HRV.hrv_x;
    HRV_Z = HRV.hrv_z;
    HRV_Y = HRV.hrv_y;
    HRV_YZ = HRV_Y./HRV_Z;
catch
    EDA_time = EDA.EDA_time;
    TEMP_time = TEMP.TEMP_time;
    HR_time = HR.HR_time;
    HRV_time = HRV.HRV_time;

    EDA_data = EDA.EDA_data;
    TEMP_data = TEMP.TEMP_data;
    HR_data = HR.HR_data;
    HRV_X = HRV.HRV_x;
    HRV_Z = HRV.HRV_z;
    HRV_Y = HRV.HRV_y;
    HRV_YZ = HRV_Y./HRV_Z;
end 

% Cut beginning if data starts with 0s 
% Find index where EDA is larger than 0.05 (smaller is invalid data). Finds the first 10 indices because sometimes
% theres will be a non zero value for 1 or 2 points and then go back to 0.
start_idxs = find(EDA_data >= 0.05, 10, 'first'); 
diff_idxs = diff(start_idxs);
i=find(diff(diff_idxs)~=0,1,'last');

if isempty(i)
    t_start = EDA_time(start_idxs(1),1);
else 
    t_start = EDA_time(start_idxs(i+1),1);
end 

% Cut end if data ends in 0 
end_idxs = find(EDA_data >= 0.05, 1, 'last'); 

if ~isempty(end_idxs)
    t_end = EDA_time(end_idxs,1);
end

[EDA_data, EDA_time] = unpad(EDA_data,EDA_time, t_start,t_end);
[TEMP_data, TEMP_time] = unpad(TEMP_data, TEMP_time, t_start,t_end);
[HR_data, HR_time] = unpad(HR_data, HR_time, t_start,t_end);
[HRV_YZ, HRV_time] = unpad(HRV_YZ,HRV_time, t_start,t_end);
[HRV_X, HRV_time] = unpad(HRV_X,HRV_time, t_start,t_end);
[HRV_Y, HRV_time] = unpad(HRV_Y,HRV_time, t_start,t_end);
[HRV_Z, HRV_time] = unpad(HRV_Z,HRV_time, t_start,t_end);

% Call preprocessing scripts
[EDA_medfilt, EDA_avefilt,EDA_interp, EDA_eurofilt] = preprocessEDA(EDA_data);
[TEMP_medfilt, TEMP_avefilt, TEMP_interp, TEMP_expfilt] = preprocessTEMP(TEMP_data);
[HR_avefilt, HR_cubic, ...
 HRV_YZ_avefilt, HRV_YZ_interp, HRV_YZ_cubic, ...
 HRV_X_avefilt, HRV_X_interp, HRV_X_cubic, ...
 HRV_Y_avefilt, HRV_Y_interp, HRV_Y_cubic, ...
 HRV_Z_avefilt, HRV_Z_interp, HRV_Z_cubic] = preprocessHR(HR_data, HR_time, HRV_YZ, HRV_X, HRV_Y, HRV_Z, HRV_time);

clean.EDA = horzcat(EDA_time,EDA_eurofilt);
clean.TEMP = horzcat(TEMP_time,TEMP_expfilt);
clean.HR = horzcat(HR_time,HR_cubic);
clean.HRVX = horzcat(HRV_time,HRV_X_cubic); 
clean.HRVY = horzcat(HRV_time,HRV_Y_cubic); 
clean.HRVZ = horzcat(HRV_time,HRV_Z_cubic); 
clean.HRVYZ = horzcat(HRV_time,HRV_YZ_cubic); 

% Plot
figure
subplot(5,1,1)
plot(unix_to_datetime(EDA_time),EDA_data(1:length(EDA_time)),'LineWidth',1)
hold on;
plot(unix_to_datetime(EDA_time),EDA_medfilt(1:length(EDA_time)),'LineWidth',1)
hold on;
plot(unix_to_datetime(EDA_time),EDA_avefilt(1:length(EDA_time)),'LineWidth',1)
hold on;
plot(unix_to_datetime(EDA_time),EDA_interp(1:length(EDA_time)),'LineWidth',1)
hold on;
plot(unix_to_datetime(EDA_time), EDA_eurofilt(1:length(EDA_time)),'LineWidth',2)
legend('raw','medfilt','avefilt','interp','eurofilt');
ylabel("EDA (us)")

subplot(5,1,2)
plot(unix_to_datetime(TEMP_time),TEMP_data(1:length(TEMP_time)),'LineWidth',1)
hold on
plot(unix_to_datetime(TEMP_time),TEMP_medfilt(1:length(TEMP_time)),'LineWidth',1)
hold on 
plot(unix_to_datetime(TEMP_time),TEMP_avefilt(1:length(TEMP_time)),'LineWidth',1)
hold on
plot(unix_to_datetime(TEMP_time),TEMP_expfilt(1:length(TEMP_time)),'LineWidth',2)
ylabel("Temperature (C)")
legend("raw", "medfilt", "avefilt", "expfilt")

subplot(5,1,3)
plot(unix_to_datetime(HR_time), HR_data(1:length(HR_time)),'LineWidth',1)
hold on 
plot(unix_to_datetime(HR_time), HR_avefilt(1:length(HR_time)),'LineWidth',1)
hold on
plot(unix_to_datetime(HR_time), HR_cubic(1:length(HR_time)),'LineWidth',2)
legend("raw", "avefilt","cubic")
ylabel("Heart rate")

subplot(5,1,4)
plot(unix_to_datetime(HRV_time), HRV_YZ(1:length(HRV_time)),'LineWidth',1)
hold on 
plot(unix_to_datetime(HRV_time), HRV_YZ_avefilt(1:length(HRV_time)),'LineWidth',1)
hold on 
plot(unix_to_datetime(HRV_time), HRV_YZ_interp(1:length(HRV_time)),'LineWidth',1)
hold on
plot(unix_to_datetime(HRV_time), HRV_YZ_cubic(1:length(HRV_time)),'LineWidth',2)
legend("raw", "avefilt", "interp","cubicspline")
ylabel("HRV Y/Z")

subplot(5,1,5)
plot(unix_to_datetime(HRV_time), HRV_Z(1:length(HRV_time)),'LineWidth',1)
hold on 
plot(unix_to_datetime(HRV_time), HRV_Z_avefilt(1:length(HRV_time)),'LineWidth',1)
hold on 
plot(unix_to_datetime(HRV_time), HRV_Z_interp(1:length(HRV_time)),'LineWidth',1)
hold on
plot(unix_to_datetime(HRV_time), HRV_Z_cubic(1:length(HRV_time)),'LineWidth',2)
legend("raw", "avefilt", "interp","cubicspline")
ylabel("HRV Z")
xlabel("Time (seconds)")

end 
