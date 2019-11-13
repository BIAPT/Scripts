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
%make_spectrogram(data,type);

%% Making the topographic map plot
%make_topographic_map(data,type);

%% Making the permutation entropy map plot
%make_permutation_entropy(data,type);

% normalized
%make_norm_permutation_entropy(data,type);

%% Making the wPLI
%make_wpli(data,type);

%% Making the dPLI
%make_dpli(data,type);

function make_dpli(data,type)
    figure;
    analysis_technique = "Alpha dPLI";
    axe1 = subplot(1,3,1);
    imagesc(data.baseline_dpli);
    colorbar;
    title(strcat(type," ",analysis_technique, " Baseline"));
    axe2 = subplot(1,3,2);
    imagesc(data.pain_dpli);
    colorbar;
    title(strcat(type," ",analysis_technique, " Hot"));
    axe3 = subplot(1,3,3);
    diff_norm_dpli = log(data.baseline_dpli ./ data.pain_dpli);
    imagesc(diff_norm_dpli);
    colorbar;
    title(strcat(type," ",analysis_technique, " Log Ratio (Baseline vs Hot)"));
    % Add in the colorbar
    colormap(axe1,'jet');
    colormap(axe2, 'jet');
    colormap(axe3, 'hot');
end

function make_wpli(data,type)
    figure;
    analysis_technique = "Alpha wPLI";
    axe1 = subplot(1,3,1);
    imagesc(data.baseline_wpli);
    colorbar;
    title(strcat(type," ",analysis_technique, " Baseline"));
    axe2 = subplot(1,3,2);
    imagesc(data.pain_wpli);
    colorbar;
    title(strcat(type," ",analysis_technique, " Hot"));
    axe3 = subplot(1,3,3);
    diff_norm_wpli = log(data.baseline_wpli ./ data.pain_wpli);
    imagesc(diff_norm_wpli);
    colorbar;
    title(strcat(type," ",analysis_technique, " Log Ratio (Baseline vs Hot)"));
    % Add in the colorbar
    colormap(axe1,'jet');
    colormap(axe2, 'jet');
    colormap(axe3, 'hot');
end

function make_norm_permutation_entropy(data,type)
    figure;
    analysis_technique = "Alpha Permutation Entropy";
    axe1 = subplot(1,3,1);
    topographic_map(data.baseline_norm_pe,data.reduced_location);
    title(strcat(type," ",analysis_technique, " Baseline"));
    axe2 = subplot(1,3,2);
    topographic_map(data.pain_norm_pe, data.reduced_location);
    title(strcat(type," ",analysis_technique, " Hot"));
    axe3 = subplot(1,3,3);
    diff_norm_pe = log(data.baseline_norm_pe ./ data.pain_norm_pe);
    topographic_map(diff_norm_pe,data.reduced_location);
    title(strcat(type," ",analysis_technique, " Log Ratio (Baseline vs Hot)"));
    % Add in the colorbar
    colormap(axe1,'copper');
    colormap(axe2, 'copper');
    colormap(axe3, 'hot');
end

function make_permutation_entropy(data,type)
    figure;
    analysis_technique = "Alpha Permutation Entropy";
    subplot(1,3,1)
    topographic_map(data.baseline_pe,data.reduced_location,'jet');
    title(strcat(type," ",analysis_technique, " Baseline"));
    subplot(1,3,2)
    topographic_map(data.pain_pe, data.reduced_location,'jet');
    title(strcat(type," ",analysis_technique, " Hot"));
    subplot(1,3,3)
    diff_pe = log(data.baseline_pe ./ data.pain_pe);
    topographic_map(diff_pe,data.reduced_location,'hot');
    title(strcat(type," ",analysis_technique, " Log Ratio (Baseline vs Hot)"));

end

function make_topographic_map(data,type)
    figure;
    analysis_technique = "Alpha Power";
    subplot(1,3,1)
    topographic_map(data.baseline_td,data.reduced_location,'jet');
    title(strcat(type," ",analysis_technique, " Baseline"));
    subplot(1,3,2)
    topographic_map(data.pain_td, data.reduced_location,'jet');
    title(strcat(type," ",analysis_technique, " Hot"));
    subplot(1,3,3)
    diff_td = log(data.baseline_td ./ data.pain_td);
    topographic_map(diff_td,data.reduced_location,'hot');
    title(strcat(type," ",analysis_technique, " Log Ratio (Baseline vs Hot)"));
end

function make_spectrogram(data,type)
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

    % the difference
    diff_spectrum = log(data.baseline_spectrum ./ data.pain_spectrum);
    figure;
    plot(data.frequencies_spectrum, diff_spectrum,'k--', 'LineWidth',2);
    grid on
    legend('Difference');
    xlabel("Frequency (Hz)");
    ylabel("Log Ratio");
    title(strcat(type," ",analysis_technique, " Log Ratio(Baseline vs Hot)"));
end

function topographic_map(data,location)
    topoplot(data,location,'maplimits','absmax', 'electrodes', 'off');
    min_color = min(data);
    max_color = max(data);
    caxis([min_color max_color])
    colorbar;
end