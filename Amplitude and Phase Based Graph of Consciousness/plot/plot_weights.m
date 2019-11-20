% Make a figure with subplot
figure
subplot(2,2,1)
bar(weights_mean_aec)
title('AEC Weights of Mean')
xlim([0,83])
ylim([-1.6,1.8])

subplot(2,2,2)
bar(weights_std_aec)
title('AEC Weights of Std')
xlim([0,83])
ylim([-1.6,1.8])

subplot(2,2,3)
bar(weights_mean_wpli)
title('wPLI Weights of Mean')
xlim([0,83])
ylim([-1.6,1.8])

subplot(2,2,4)
bar(weights_std_wpli)
title('wPLI Weights of Std')
xlim([0,83])
ylim([-1.6,1.8])