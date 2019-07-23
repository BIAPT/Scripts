full_features = [weights_mean weights_std];

% Taking the absolute value and then hard normalized between 0 and 1
full_features = abs(full_features);
full_features = (full_features - min(full_features)) ./ (max(full_features) - min(full_features));

weights_mean_normalized = full_features(1:82);
weights_std_normalized = full_features(83:164);
