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


% TODO : CONTINUE THE ANALYSIS!
% FP wPLI
avg_left_lateral_wpli = wsas.result_wpli.data.avg_left_lateral_wpli';
avg_left_midline_wpli = wsas.result_wpli.data.avg_left_midline_wpli';
avg_right_lateral_wpli = wsas.result_wpli.data.avg_right_lateral_wpli';
avg_right_midline_wpli = wsas.result_wpli.data.avg_right_midline_wpli';

figure;
min_point = min([avg_left_midline_wpli, avg_left_lateral_wpli, avg_right_midline_wpli, avg_right_lateral_wpli]);
max_point = max([avg_left_midline_wpli, avg_left_lateral_wpli, avg_right_midline_wpli, avg_right_lateral_wpli]);

subplot(2,2,1);
x_index = linspace(0,total_x,length(avg_left_midline_wpli));
plot(x_index, avg_left_midline_wpli, 'LineWidth',3);
title(strcat(title_name, ' wPLI Left Midline'));
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,2);
x_index = linspace(0,total_x,length(avg_right_midline_wpli));
plot(x_index, avg_right_midline_wpli, 'LineWidth',3);
title(strcat(title_name, ' wPLI Right Midline'));
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,3);
x_index = linspace(0,total_x,length(avg_left_lateral_wpli));
plot(x_index, avg_left_lateral_wpli, 'LineWidth',3);
title(strcat(title_name, ' wPLI Left Lateral'));
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,4);
x_index = linspace(0,total_x,length(avg_right_lateral_wpli));
plot(x_index, avg_right_lateral_wpli, 'LineWidth',3);
title(strcat(title_name, ' wPLI Right Lateral'));
ylim([min_point,max_point]);
xlabel('Seconds');


% FP dPLI
avg_left_lateral_dpli = wsas.result_dpli.data.avg_left_lateral_dpli';
avg_left_midline_dpli = wsas.result_dpli.data.avg_left_midline_dpli';
avg_right_lateral_dpli = wsas.result_dpli.data.avg_right_lateral_dpli';
avg_right_midline_dpli = wsas.result_dpli.data.avg_right_midline_dpli';

figure;
min_point = min([avg_left_midline_dpli avg_left_lateral_dpli, avg_right_midline_dpli, avg_right_lateral_dpli]);
max_point = max([avg_left_midline_dpli avg_left_lateral_dpli, avg_right_midline_dpli, avg_right_lateral_dpli]);

subplot(2,2,1);
x_index = linspace(0,total_x,length(avg_left_midline_dpli));
plot(x_index, avg_left_midline_dpli, 'LineWidth',3);
title(strcat(title_name, ' dPLI Left Midline'));
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,2);
x_index = linspace(0,total_x,length(avg_right_midline_dpli));
plot(x_index, avg_right_midline_dpli, 'LineWidth',3);
title(strcat(title_name, 'dPLI Right Midline'));
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,3);
x_index = linspace(0,total_x,length(avg_left_lateral_dpli));
plot(x_index, avg_left_lateral_dpli, 'LineWidth',3);
title(strcat(title_name, ' dPLI Left Lateral'));
ylim([min_point,max_point]);
xlabel('Seconds');

subplot(2,2,4);
x_index = linspace(0,total_x,length(avg_right_lateral_dpli));
plot(x_index, avg_right_lateral_dpli, 'LineWidth',3);
title(strcat(title_name, ' dPLI Right Lateral'));
ylim([min_point,max_point]);
xlabel('Seconds');

% Hub Location
figure
hub_relative_position = result_hl.data.hub_normalized_value;
x_index = linspace(0,total_x,length(hub_relative_position));
plot(x_index, hub_relative_position, 'LineWidth', 3);
title(strcat(title_name, ' HL Relative Position'));
xlabel('Seconds');


% Permutation Entropy
figure;
avg_permutation_entropy_anterior = result_pe.data.avg_permutation_entropy_anterior;
avg_permutation_entropy_posterior = result_pe.data.avg_permutation_entropy_posterior;
min_point = min([avg_permutation_entropy_anterior, avg_permutation_entropy_posterior]);
max_point = max([avg_permutation_entropy_anterior, avg_permutation_entropy_posterior]);

subplot(2,1,1);
x_index = linspace(0,total_x,length(avg_permutation_entropy_anterior));
plot(x_index, avg_permutation_entropy_anterior, 'LineWidth', 3);
title(strcat(title_name, ' PE Anterior'));
ylim([min_point, max_point]);
xlabel('Seconds');

subplot(2,1,2);
x_index = linspace(0,total_x,length(avg_permutation_entropy_posterior));
plot(x_index, avg_permutation_entropy_posterior, 'LineWidth', 3);
title(strcat(title_name, ' PE Parietal'));
ylim([min_point, max_point]);
xlabel('Seconds');