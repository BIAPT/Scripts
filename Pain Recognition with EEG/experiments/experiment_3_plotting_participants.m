%{
    This script was written by Yacine Mahdid 2019-11-10 for the analysis of
    the Pain EEG data collected from the Shrinner hospital.
    Here we are using no_pain and hot1 values
%}
%% Make a script to plot the healthy participants folder
% Setting up path variables
result_path = "";

type = 'Healthy Participant';

data = load(strcat(result_path,'HEAVG.mat'));
data = data.result;

%% Making the Spectrogram plot
analysis_technique = 'Spectrogram';

figure;
plot(data.frequencies_spectrum,data.baseline_spectrum,'b--', 'LineWidth',2);
hold on
plot(data.frequencies_spectrum, data.pain_spectrum,'r--', 'LineWidth',2);
grid on
legend('Rest','Hot');
xlabel("Frequency (Hz)");
ylabel("Power (dB)");
title(strcat(type," ",analysis_technique, " Baseline vs Hot"));
