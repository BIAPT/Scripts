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
