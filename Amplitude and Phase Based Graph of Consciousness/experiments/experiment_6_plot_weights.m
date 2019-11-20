%% Variables initialization
%% Experiment variable
analysis = 'aec'; % aec or wpli
epoch = 'emf5'; % emf5 or eml5
side = 'both'; % left right or both

% Global variables
data_folder = 'weights_data/';
top = 0.2; % We can play around this

% Colormap [nothing = 1, mean = 2, std = 3, both= 4]
color_map = [ 1 1 1 ; 0.9290, 0.6940, 0.1250; 0, 0.4470, 0.7410 ; 0.4660, 0.6740, 0.1880]; % This is the colormap for the not important, mean, std, both

% Side of brain to map (left or right)
if(strcmp(side,'left'))
   roi_range = 1:41;
elseif(strcmp(side,'right'))
    roi_range = 42:82;
else
    roi_range = 1:82;
end 

% Load the necessary variables
load('../plot/ROI_MNI_V5_nii.mat'); % AALreg = volume image of 120 ROIs
load('../plot/ROI_MNI_V5_vol.mat'); % ROI1 = contains info on ROIs
load('../plot/Atlas_scouts.mat'); % scout = info of edited atlas

% Set the scouts name and labels
scout = scout(roi_range);

number_locs = length(scout); % number_locs =  82

% Matches scout ROIs to original AAL atlas
labellist = zeros(1,number_locs);

for jj = 1:length(scout)
        ll = scout(jj).Label;
        ll(end-1)='_';
    for ii = 1:length(ROI1)
        if strcmp(ROI1(ii).Nom_L,ll)
            labellist(jj) = ii;
        end
    end
end

% Fix the ROI
roi = zeros([size(AALreg),number_locs]);
for reg = 1:number_locs
    roi(:,:,:,reg) = AALreg == ROI1(labellist(reg)).ID;
end

% Load the data
filename = strcat(data_folder,analysis,'_',epoch,'_weights.mat');
data = load(filename);
abs_weights = abs(data.weights);
norm_weights = (abs_weights - min(abs_weights)) ./ (max(abs_weights) - min(abs_weights));
t_weights = threshold(top, norm_weights);

%% Weights vector initialization 

% Side of brain to map (left or right)
if(strcmp(side,'left'))
   roi_range = 1:41;
elseif(strcmp(side,'right'))
    roi_range = 42:82;
else
    roi_range = 1:82;
end 
%% Plot the weights
plot_weights_bar(analysis, norm_weights)

%% Plot the brain with the correct weights
% Can specify colour limits, here 0 to 1
aal_brain(roi_range,t_weights,roi,color_map);
camlight (30,0)
set(gca,'view',[270 90]); %Changes angle of view




%% Helper function
function aal_brain(roi_range,weights,roi,color_map)
    % fc = functional connectivity matrix or vector
    % roi = volume matrix of ROI regions, 3dims x no. ROIs  
    
    % Create the figure
    figure
    axis ([0,100,0,50,0,100])
    c = colormap(color_map);
    hold all

    colorindex = weights(roi_range);

    for reg = 1:length(roi_range)
        roisurf=isosurface(roi(:,:,:,reg),0.5);
        h = trisurf(roisurf.faces,roisurf.vertices(:,1),roisurf.vertices(:,2),roisurf.vertices(:,3));
        set(h,'facecolor',c(colorindex(reg),:),'facealpha',0.8,'LineWidth',0.1,'LineStyle','none');
    end

    set(gca,'view',[-90 90])
    axis equal
    axis off
    set(gcf,'color','white')
end

function [weights] = threshold(top, weights)
    all_weights = sort(weights);
    top_index = floor(length(all_weights)*(1-top));
    weight_threshold = all_weights(top_index);
    
    % Get the mean and std out of the weights vector
    weights_mean = weights(1:82);
    weights_std = weights(83:end);
    
    % Treshold the values
    weights_mean(weights_mean < weight_threshold) = 0;
    weights_mean(weights_mean >= weight_threshold) = 1;
    weights_std(weights_std < weight_threshold) = 0;
    weights_std(weights_std >= weight_threshold) = 1;

    % Need to set the weights like this:
    % [nothing = 1, mean = 2, std = 3, both= 4]
    weights = zeros(1,length(weights_mean));
    
    for i = 1:length(weights)
       if(weights_mean(i) && weights_std(i))
           weights(i) = 4;
       elseif(weights_mean(i))
           weights(i) = 2;
       elseif(weights_std(i))
           weights(i) = 3;
       else
           weights(i) = 1;
       end
    end
end