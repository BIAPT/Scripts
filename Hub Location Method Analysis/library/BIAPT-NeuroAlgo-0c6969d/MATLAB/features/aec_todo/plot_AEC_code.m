
fband=4;
R=82;

switch fband
    
    case 1
        lowpass = 4;
        highpass = 1;
        band = 'delta'
        d = 10; % Downsampling factor
    case 2
        lowpass = 8;
        highpass = 4;
        band = 'theta'
        d = 10;
    case 3
        lowpass = 13;
        highpass = 8;
        band = 'alpha'
        d = 5;
    case 4
        lowpass = 30;
        highpass = 13;
        band = 'beta'
        d = 5;
    case 5
        lowpass = 48;
        highpass = 30;
        band = 'gamma'
        d = 4;
end
% Bandwidth
B = lowpass-highpass;
% Window duration for PLI calculation
T = 100/(2*B);    % ~100 effective points

PLI_files=dir(strcat('H:/new_analysis/PLI_OUT/', band , '/PLI*'));
AEC_files=dir(strcat('H:/new_analysis/AEC_OUT/', band, '/AEC*'));

for j=1:10
    fp=PLI_files(j,1).name;
    fa=AEC_files(j,1).name;
    fp=fp(1:end-5);
    fa=fa(1:end-5);
    fp= strcat('H:/figures_AECPLI/', band, '/', fp);
    fa= strcat('H:/figures_AECPLI/', band, '/', fa);
    
    
    load(strcat('H:/new_analysis/PLI_OUT/', band,'/' ,PLI_files(j,1).name));
    load(strcat('H:/new_analysis/AEC_OUT/', band,'/' ,AEC_files(j,1).name));

    PLI=((PLI_OUT{1,1} + PLI_OUT{1,2}+ PLI_OUT{1,3}+ PLI_OUT{1,4}+ PLI_OUT{1,5}+ PLI_OUT{1,6}+ PLI_OUT{1,7}+ PLI_OUT{1,8}+PLI_OUT{1,9})/9);
    AEC=((AEC_OUT{1,1} + AEC_OUT{1,2}+ AEC_OUT{1,3}+ AEC_OUT{1,4}+ AEC_OUT{1,5}+ AEC_OUT{1,6}+ AEC_OUT{1,7}+ AEC_OUT{1,8}+AEC_OUT{1,9})/9);
    
    
    h=figure('position', [200, 100, 900, 800])
    %imagesc(AEC); colorbar; caxis([min(AEC(~eye(R))), max(AEC(~eye(R)))]);
    imagesc(AEC); colorbar; caxis([0.05, 0.18]);
    colormap jet; axis off
    %title('AEC, multivariate correction')
    
    
    hd=figure('position', [100, 50, 900, 800]);
    %imagesc(PLI); colorbar; caxis([min(PLI(~eye(R))), max(PLI(~eye(R)))]);
    imagesc(PLI); colorbar; caxis([0.07,0.13]);
    colormap jet; axis off
    %title(sprintf('PLI, with %.1fs windows',T))
    
    
    
    
    savefig(h, strcat(fa, '.fig'));
    savefig(hd, strcat(fp, '.fig'));
    print(h,strcat(fa, '.tiff') ,'-dtiff')
    print(hd,strcat(fp, '.tiff') ,'-dtiff')
    close all;
end