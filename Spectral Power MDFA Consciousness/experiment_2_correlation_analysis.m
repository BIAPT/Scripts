% Yacine Mahdid 2020-01-10
% This script was made for the revision of the motif analysis paper, where
% we want to compare alpha topographic map for each participant version the
% motif frequency topographic map. Here we are doing a correlation
% analysis at three level :
% 1) all participant, channels and condition pooled in a scatterplot
% 2) all participant, channels and condition pooled in a Hexagonal binned
% scatterplot
% 3) average participants in a scatter plot

states = {'BASELINE', 'IF5', 'EMF5','EML30','EML10','EML5','RECOVERY'};
participants = {'MDFA03','MDFA05','MDFA06','MDFA07','MDFA11','MDFA12','MDFA15'};
m_id = 7; % 7


motif_frequency = [];
power = [];
for p = 1:length(participants)
    participant = participants{p};
    
    % Load the power
    input_path_power = strcat('E:/research projects/motif_analysis/power/',participant,'.mat');
    data = load(input_path_power);
    power_state_data = data.state_data;
    
    % Load the motif frequency (TO CHANGE HERE)
    input_path_power = strcat('E:/research projects/motif_analysis/motif/',participant,'.mat');
    data = load(input_path_power);
    motif_state_data = data.state_data;
    
    for i = 1:length(states)
        power = cat(2,power,power_state_data(i).power);
        motif_frequency = cat(2,motif_frequency, motif_state_data(i).frequency); 
    end
end

%% Loading the pre-computed datapoints for each participants
% Here what we will do is make two huge vector for each participant that
% will be of the same size for the power spectra and for the motif
% frequency

%% Create a scatter plot 
% Calculate first the correlation coefficient
motif_frequency = motif_frequency(m_id,:);
[R,~] = corrcoef(motif_frequency,power);% This gives a matrix of correlation
correlation = R(2); % We just take the correlation between frequency and power and not power with power
figure;
scatter(motif_frequency,power);
xlabel('Motif Frequency');
ylabel('Power')
title(strcat("Power versus Motif ",string(m_id)," Frequency at Alpha (R = ",string(correlation),")"));

%% Write to CSV for the hexagonal binning
fid = fopen("motif_power.csv", 'wt');
fprintf(fid, '%s,%s\n', "motif_frequency","power");
for i = 1:length(motif_frequency)
    fprintf(fid, '%f,%f\n', motif_frequency(i), power(i));    
end
fclose(fid);
