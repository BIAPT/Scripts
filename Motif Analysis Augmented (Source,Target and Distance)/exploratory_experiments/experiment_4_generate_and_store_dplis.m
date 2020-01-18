%{
    Script written by Yacine Mahdid
    The goal of this script is to load the right dataset generate the right
    dPLI matrices and save them to the file system to be feed to the next experiment
    that will involve create the motifs.

    The choosen participant was MDFA17 as proposed by Danielle Nadin
    The timepoints are:
    baseline, induction, unconscious, - 30min, -10min, -5min, +30min,
    +180min
%}

%% Variables Initalization
data_location = 'C:\Users\biapt\Desktop\motif fix\mdfa17_data';
output_location = 'C:\Users\biapt\Desktop\motif fix\mdfa17_dpli_data\';
participant = 'MDFA17';
epochs = {'BASELINE'}; %,'IF5', 'EMF5', 'EML30', 'EML10', 'EML5', 'RECOVERY'};

% Experiment variables
% for dPLI
frequency_band = [8 13]; % This is in Hz
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;

%% Iterating over the files
for e_i = 1:length(epochs)
    % Load the data
    filename = strcat(participant,'_',epochs{e_i},'.set');
    recording = load_set(filename,data_location);
    % Calculate the dPLI
    result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
    output_filename = strcat(output_location,participant,'_',epochs{e_i},'_dpli','.mat');
    save(output_filename,'result_dpli');
end




