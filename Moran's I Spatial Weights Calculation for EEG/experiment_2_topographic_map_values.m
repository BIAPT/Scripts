%% Script to check how to modify the return grid_or_val to increase its size
% Conclusion: its gridscale
i = 10;

figure; 
title(strcat('3D euclidean distance for  ', channel_location(i).labels)); 
[h grid_or_val plotrad_or_grid, xmesh, ymesh] = topoplot(weight_matrix(i,:), channel_location,'gridscale',129); 
colorbar;