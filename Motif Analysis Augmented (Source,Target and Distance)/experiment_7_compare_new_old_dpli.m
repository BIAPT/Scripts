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
epochs = {'BASELINE'};

% Load the dplis
old_dpli = z_score;
new_dpli = result_dpli.data.avg_dpli;

% Plot the figures
figure;
subplot(2,1,1)
imagesc(old_dpli);
colormap('jet');
colorbar
subplot(2,1,2)
imagesc(new_dpli);
colorbar


% Make the contrasts matrix
figure
imagesc(old_dpli-new_dpli);
colormap('hot');
colorbar;

%{
    The outcome of this is that the dpli is pretty much the same, which
    means that the dPLI calcualte by NA is the same as the one calculated
    by EEGapp.
    -> This should means that if we calculate motif using the dPLI with NA
    we should directly be able to get the same result than we have in the
    motif paper.
%}
