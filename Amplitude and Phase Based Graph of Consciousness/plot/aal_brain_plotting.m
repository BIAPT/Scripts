%% Variables initialization
% Weights
data = load('weights.mat');
weights_pli_pre_ROC = data.weights_pli_pre_ROC;
weights_aec_pre_ROC = data.weights_aec_pre_ROC;
weights_pli_unconscious = data.weights_pli_unconscious;
weights_aec_unconscious = data.weights_aec_unconscious;

% Colormap [nothing = 1, mean = 2, std = 3, both= 4]
color_map = [ 1 1 1 ; 0.4660, 0.6740, 0.1880 ; 0.9290, 0.6940, 0.1250; 0, 0.4470, 0.7410]; % This is the colormap for the not important, mean, std, both

%% Weights vector initialization and USER INPUT
weights = weights_aec_unconscious; %To change using the values above
% Side of brain to map (left or right)
side = 'right';
if(strcmp(side,'left'))
   roi_range = 1:41;
elseif(strcmp(side,'right'))
    roi_range = 42:82;
end

%% Load the necessary variables
load('ROI_MNI_V5_nii.mat'); % AALreg = volume image of 120 ROIs
load('ROI_MNI_V5_vol.mat'); % ROI1 = contains info on ROIs
load('Atlas_scouts.mat'); % scout = info of edited atlas 

%% Set the scouts name and labels
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

%% Fix the ROI
roi = zeros([size(AALreg),number_locs]);
for reg = 1:number_locs
    roi(:,:,:,reg) = AALreg == ROI1(labellist(reg)).ID;
end

%% Plot the brain with the correct weights
% Can specify colour limits, here 0 to 1
aal_brain(roi_range,weights,roi,color_map);
camlight (90,30)
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