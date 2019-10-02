legend_tags = ['Baseline'; 'Sedation'; 'Recovery'];
wsas = [load('wsas09_baseline.mat'), load('wsas09_sedation.mat'), load('wsas09_recovery.mat')];

sampling_rate = wsas(1).result_wpli.metadata.sampling_rate;
number_points = 75001; % got that from the recording class (assigned an issue ot have it in result)

%% Plot the Results
total_x = number_points/sampling_rate; % 150 recording of 5 seconds each

% Spectral Power Ratio
ratio_alpha_theta = [];
ratio_beta_alpha = [];
total = [];
for i = 1:length(wsas)
    ratio_alpha_theta = [ratio_alpha_theta; wsas(i).result_spr.data.ratio_alpha_theta];
    ratio_beta_alpha = [ratio_beta_alpha; wsas(i).result_spr.data.ratio_beta_alpha];
    total = [total, wsas(i).result_spr.data.ratio_alpha_theta, wsas(i).result_spr.data.ratio_beta_alpha];
end

figure;
min_point = min(total);
max_point = max(total);

subplot(2,1,1);
x_index = linspace(0,total_x,length(ratio_alpha_theta));
for i = 1:length(wsas)
    hold on
    plot(x_index, ratio_alpha_theta(i,:), 'LineWidth', 3);
    hold off
end
title(' SPR Alpha Theta');
ylim([min_point, max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,1,2);
x_index = linspace(0,total_x,length(ratio_beta_alpha));
for i = 1:length(wsas)
    hold on
    plot(x_index,ratio_beta_alpha(i,:), 'LineWidth', 3);
    hold off
end
title(' SPR Beta Alpha');
ylim([min_point, max_point]);
xlabel('Seconds');
legend(legend_tags);

% Topographic Distribution

avg_power_ratio_front_posterior = [];
total = [];
for i = 1:length(wsas)
    avg_power_ratio_front_posterior = [avg_power_ratio_front_posterior; wsas(i).result_td.data.avg_power_ratio_front_posterior];
    total = [total, wsas(i).result_td.data.avg_power_ratio_front_posterior]
end

figure;
min_point = min(total);
max_point = max(total);
x_index = linspace(0,total_x,length(avg_power_ratio_front_posterior));
for i = 1:length(wsas)
    hold on
    plot(x_index, avg_power_ratio_front_posterior(i,:), 'LineWidth', 3);
    hold off
end
title(" TD ratio Front vs Back 10Hz");
xlabel('Seconds');
legend(legend_tags);

% Phase Amplitude Coupling
rpt_anterior = [];
rpt_posterior = [];
total = [];
for i = 1:length(wsas)
    rpt_anterior = [rpt_anterior; wsas(i).result_pac.data.ratio_peak_through_anterior];
    rpt_posterior = [rpt_posterior; wsas(i).result_pac.data.ratio_peak_through_posterior];
    total = [total, wsas(i).result_pac.data.ratio_peak_through_anterior, wsas(i).result_pac.data.ratio_peak_through_posterior]
end

figure;
min_point = min(total)
max_point = max(total);

subplot(2,1,1);
x_index = linspace(0,total_x,length(rpt_anterior));
for i = 1:length(wsas)
    hold on
    plot(x_index, rpt_anterior(i,:), 'LineWidth', 3);
    hold off
end
title(' PAC RPT Frontal');
ylim([min_point, max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,1,2);
x_index = linspace(0,total_x,length(rpt_posterior));
for i = 1:length(wsas)
    hold on
    plot(x_index, rpt_posterior(i,:), 'LineWidth', 3);
    hold off
end
title(' PAC RPT Parietal');
ylim([min_point, max_point]);
xlabel('Seconds');
legend(legend_tags);



% FP wPLI

left_lateral_wpli = [];
left_midline_wpli = [];
right_lateral_wpli = [];
right_midline_wpli = [];
total = [];

for i = 1:length(wsas)
    data = wsas(i).result_wpli.data;
    left_lateral_wpli = [left_lateral_wpli; data.avg_left_lateral_wpli'];
    left_midline_wpli = [left_midline_wpli; data.avg_left_midline_wpli'];
    right_lateral_wpli = [right_lateral_wpli; data.avg_right_lateral_wpli'];
    right_midline_wpli = [right_midline_wpli; data.avg_right_midline_wpli'];
    total = [total, data.avg_left_lateral_wpli', data.avg_left_midline_wpli', data.avg_right_lateral_wpli', data.avg_right_midline_wpli'];
end

figure;
min_point = min(total);
max_point = max(total);

subplot(2,2,1);
x_index = linspace(0,total_x,length(left_midline_wpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, left_midline_wpli(i,:), 'LineWidth',3);
    hold off
end
title(' Alpha wPLI Left Midline');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,2,2);
x_index = linspace(0,total_x,length(right_midline_wpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, right_midline_wpli(i,:), 'LineWidth',3);
    hold off
end
title(' wPLI Right Midline');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,2,3);
x_index = linspace(0,total_x,length(left_lateral_wpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, left_lateral_wpli(i,:), 'LineWidth',3);
    hold off
end
title(' Alpha wPLI Left Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,2,4);
x_index = linspace(0,total_x,length(right_lateral_wpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, right_lateral_wpli(i,:), 'LineWidth',3);
    hold off
end
title(' Alpha wPLI Right Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);


% FP dPLI
left_lateral_dpli = [];
left_midline_dpli = [];
right_lateral_dpli = [];
right_midline_dpli = [];
total = [];

for i = 1:length(wsas)
    data = wsas(i).result_dpli.data;
    left_lateral_dpli = [left_lateral_dpli; data.avg_left_lateral_dpli'];
    left_midline_dpli = [left_midline_dpli; data.avg_left_midline_dpli'];
    right_lateral_dpli = [right_lateral_dpli; data.avg_right_lateral_dpli'];
    right_midline_dpli = [right_midline_dpli; data.avg_right_midline_dpli'];
    total = [total, data.avg_left_lateral_dpli', data.avg_left_midline_dpli', data.avg_right_lateral_dpli', data.avg_right_midline_dpli'];
end

figure;
min_point = min(total);
max_point = max(total);

subplot(2,2,1);
x_index = linspace(0,total_x,length(left_midline_dpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, left_midline_dpli(i,:), 'LineWidth',3);
    hold off
end
title(' Alpha dPLI Left Midline');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,2,2);
x_index = linspace(0,total_x,length(right_midline_dpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, right_midline_dpli(i,:), 'LineWidth',3);
    hold off
end
title(' Alpha dPLI Right Midline');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,2,3);
x_index = linspace(0,total_x,length(left_lateral_dpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, left_lateral_dpli(i,:), 'LineWidth',3);
    hold off
end
title(' Alpha dPLI Left Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,2,4);
x_index = linspace(0,total_x,length(right_lateral_dpli));
for i = 1:length(wsas)
    hold on
    plot(x_index, right_lateral_dpli(i,:), 'LineWidth',3);
    hold off
end
title(' Alpha dPLI Right Lateral');
ylim([min_point,max_point]);
xlabel('Seconds');
legend(legend_tags);

% Hub Location

hub_relative_position = [];
total = [];
for i = 1:length(wsas)
    data = wsas(i).result_hl.data;
    hub_relative_position = [hub_relative_position; data.hub_normalized_value];
    total = [total, data.hub_normalized_value];
end
min_point = min(total);
max_point = max(total);

figure
x_index = linspace(0,total_x,length(hub_relative_position));
for i = 1:length(wsas)
    hold on
    plot(x_index, hub_relative_position(i,:), 'LineWidth', 3);
    hold off
end
ylim([min_point, max_point]);
title(' Alpha HL Relative Position');
xlabel('Seconds');
legend(legend_tags);


% Permutation Entropy

pe_anterior = [];
pe_posterior = [];
total = [];
for i = 1:length(wsas)
    data = wsas(i).result_pe.data;
    pe_anterior = [pe_anterior; data.avg_permutation_entropy_anterior];
    pe_posterior = [pe_posterior; data.avg_permutation_entropy_posterior];
    total = [total, data.avg_permutation_entropy_anterior, data.avg_permutation_entropy_posterior]
end

min_point = min(total);
max_point = max(total);

figure
subplot(2,1,1);
x_index = linspace(0,total_x,length(pe_anterior));
for i = 1:length(wsas)
    hold on
    plot(x_index, pe_anterior(i,:), 'LineWidth', 3);
    hold off
end
title(' PE Anterior');
ylim([min_point, max_point]);
xlabel('Seconds');
legend(legend_tags);

subplot(2,1,2);
x_index = linspace(0,total_x,length(pe_posterior));
for i = 1:length(wsas)
    hold on
    plot(x_index, pe_posterior(i,:), 'LineWidth', 3);
    hold off
end
title(' PE Posterior');
ylim([min_point, max_point]);
xlabel('Seconds');
legend(legend_tags);