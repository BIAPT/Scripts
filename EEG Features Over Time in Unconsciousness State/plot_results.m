participant = "wsas09";
condition = "sedation";
title_name = strcat(participant, " ", condition);
name = strcat(participant,'_',condition);
filename = strcat(name,'.mat');
wsas = load(filename);

sampling_rate = wsas.result_wpli.metadata.sampling_rate;
number_points = 75001; % got that from the recording class (assigned an issue ot have it in result)

%% Plot the Results
total_x = number_points/sampling_rate; % 150 recording of 5 seconds each

% Spectral Power Ratio
ratio_alpha_theta = wsas.result_spr.data.ratio_alpha_theta;
ratio_beta_alpha = wsas.result_spr.data.ratio_beta_alpha;

figure;
min_point = min([ratio_alpha_theta, ratio_beta_alpha]);
max_point = max([ratio_alpha_theta, ratio_beta_alpha]);

subplot(2,1,1);
x_index = linspace(0,total_x,length(ratio_alpha_theta));
plot(x_index, ratio_alpha_theta, 'LineWidth', 3);
title(strcat(title_name,' SPR Alpha Theta'));
ylim([min_point, max_point]);
xlabel('Seconds');

subplot(2,1,2);
x_index = linspace(0,total_x,length(ratio_beta_alpha));
plot(x_index,ratio_beta_alpha, 'LineWidth', 3);
title(strcat(title_name, ' SPR Beta Alpha'));
ylim([min_point, max_point]);
xlabel('Seconds');


% Topographic Distribution
avg_power_ratio_front_posterior = wsas.result_td.data.avg_power_ratio_front_posterior;
figure;
x_index = linspace(0,total_x,length(avg_power_ratio_front_posterior));
plot(x_index, avg_power_ratio_front_posterior, 'LineWidth', 3);
title(strcat(title_name," TD ratio Front vs Back"));
xlabel('Seconds');

% Phase Amplitude Coupling
ratio_peak_through_anterior = wsas.result_pac.data.ratio_peak_through_anterior;
ratio_peak_through_posterior = wsas.result_pac.data.ratio_peak_through_posterior;

figure;
min_point = min([ratio_peak_through_anterior, ratio_peak_through_posterior]);
max_point = max([ratio_peak_through_anterior, ratio_peak_through_posterior]);

subplot(2,1,1);
x_index = linspace(0,total_x,length(ratio_peak_through_anterior));
plot(x_index, ratio_peak_through_anterior, 'LineWidth', 3);
title(strcat(title_name,' PAC RPT Frontal'));
ylim([min_point, max_point]);
xlabel('Seconds');

subplot(2,1,2);
x_index = linspace(0,total_x,length(ratio_peak_through_posterior));
plot(x_index, ratio_peak_through_posterior, 'LineWidth', 3);
title(strcat(title_name,' PAC RPT Parietal'));
ylim([min_point, max_point]);
xlabel('Seconds');

return;
% TODO : CONTINUE THE ANALYSIS!
% FP wPLI
avg_left_lateral_wpli = wsas.result_wpli.data.avg_left_lateral_wpli;
avg_left_midline_wpli = wsas.result_wpli.data.avg_left_midline_wpli;
avg_right_lateral_wpli = wsas.result_wpli.data.avg_right_lateral_wpli;
avg_right_midline_wpli = wsas.result_wpli.data.avg_right_midline_wpli;

figure;
min_point = min([avg_left_midline_wpli avg_left_lateral_wpli, avg_right_midline_wpli, avg_right_lateral_wpli]);
max_point = max([avg_left_midldine_wpli avg_left_lateral_wpli, avg_right_midline_wpli, avg_right_lateral_wpli]);

subplot(2,2,1);
x_index = linspace(0,total_x,length(fp_wpli.left_midline));
plot(x_index, fp_wpli.left_midline, 'LineWidth',3);
title('wPLI Left Midline');
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,2);
x_index = linspace(0,total_x,length(fp_wpli.right_midline));
plot(x_index, fp_wpli.right_midline, 'LineWidth',3);
title('wPLI Right Midline');
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,3);
x_index = linspace(0,total_x,length(fp_wpli.left_lateral));
plot(x_index, fp_wpli.left_lateral, 'LineWidth',3);
title('wPLI Left Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,4);
x_index = linspace(0,total_x,length(fp_wpli.right_lateral));
plot(x_index, fp_wpli.right_lateral, 'LineWidth',3);
title('wPLI Right Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');

% FP dPLI
figure;
min_point = min([fp_dpli.left_midline fp_dpli.left_lateral, fp_dpli.right_midline, fp_dpli.right_lateral]);
max_point = max([fp_dpli.left_midline fp_dpli.left_lateral, fp_dpli.right_midline, fp_dpli.right_lateral]);

subplot(2,2,1);
x_index = linspace(0,total_x,length(fp_dpli.left_midline));
plot(x_index, fp_dpli.left_midline, 'LineWidth',3);
title('dPLI Left Midline');
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,2);
x_index = linspace(0,total_x,length(fp_dpli.right_midline));
plot(x_index, fp_dpli.right_midline, 'LineWidth',3);
title('dPLI Right Midline');
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,3);
x_index = linspace(0,total_x,length(fp_dpli.left_lateral));
plot(x_index, fp_dpli.left_lateral, 'LineWidth',3);
title('dPLI Left Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,4);
x_index = linspace(0,total_x,length(fp_dpli.right_lateral));
plot(x_index, fp_dpli.right_lateral, 'LineWidth',3);
title('dPLI Right Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');

% Hub Location
figure;
x_index = linspace(0,total_x,length(hl_relative_position));
plot(x_index, hl_relative_position, 'LineWidth', 3);
title('HL Relative Position');
xlabel('Seconds');

% Permutation Entropy
figure;
min_point = min([pe_frontal, pe_parietal]);
max_point = max([pe_frontal, pe_parietal]);

subplot(2,1,1);
x_index = linspace(0,total_x,length(pe_frontal));
plot(x_index, pe_frontal, 'LineWidth', 3);
title('PE Frontal');
ylim([min_point, max_point]);
xlabel('Seconds');

subplot(2,1,2);
x_index = linspace(0,total_x,length(pe_parietal));
plot(x_index, pe_parietal, 'LineWidth', 3);
title('PE Parietal');
ylim([min_point, max_point]);
xlabel('Seconds');