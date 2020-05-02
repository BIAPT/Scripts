% Function to input: 
% imagepath/or image itself
lower=[0.3 0.35 0.4 0.45];   % = lower boundaries of colorbar
upper=[0.7 0.65 0.6 0.55];   % = upper boundaries of colorbar               
electrodes=[15,15,8,3,5];    % = number of electrodes in [F,C,P,O,T]
imagepath="baseline_left_dpli.fig";
nr = 1;


modify_images_version2(imagepath,lower,upper,electrodes,nr)

