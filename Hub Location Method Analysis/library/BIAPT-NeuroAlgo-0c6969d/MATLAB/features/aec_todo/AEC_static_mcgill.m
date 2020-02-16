% Lucrezia Liuzzi, SPMIC, University of Nottingham, 09/06/2017
%
% Comparison of amplitude envelope correlation (AEC) 
% (1) with multivariate leakage correction applied over all time course, 
% (2) multivariate leakage correction applied in sliding windows,
% (3) pairwise correction in sliding windowns ,
% and phase lag index (PLI) averaged over shorter sliding windows.
%
% Requires "symmetric_orthogonalise.m", "leakage_reduction.mexa64", and
% EEGlab package or alternative frequency filter (see line 72).

clear all
close all
clc



%% Load data

load MDFA05_eyesclosed_1.mat

Value= Value([82 62 54 56 58 60 30 26 34 32 28 24 36 86 66 76 84 74 72 70 88 3 78 52 50 48 5 22 46 38 40 98 92 90 96 94 68 16 18 20 44 83 63 55 57 59 61 31 27 35 33 29 25 37 87 67 77 85 75 71 73 89 4 79 53 51 49 6 23 47 39 41 99 93 91 97 95 69 17 19 21 45],:);
Atlas.Scouts = Atlas.Scouts([82 62 54 56 58 60 30 26 34 32 28 24 36 86 66 76 84 74 72 70 88 3 78 52 50 48 5 22 46 38 40 98 92 90 96 94 68 16 18 20 44 83 63 55 57 59 61 31 27 35 33 29 25 37 87 67 77 85 75 71 73 89 4 79 53 51 49 6 23 47 39 41 99 93 91 97 95 69 17 19 21 45]);
% Get ROI labels from atlas
LABELS = cell(1,82);
for ii = 1:82
    LABELS{ii} = Atlas.Scouts(ii).Label;
end

% Sampling frequency
f = 1/(Time(2)-Time(1));


%%  Choose frequency band
fband = 4;
switch fband
    
    case 1
        lowpass = 4;
        highpass = 1;
        fname = 'delta';
        d = 10; % Downsampling factor
    case 2
        lowpass = 8;
        highpass = 4;
        fname = 'theta';
        d = 10;
    case 3
        lowpass = 13;
        highpass = 8;
        fname = 'alpha';
        d = 5;
    case 4
        lowpass = 30;
        highpass = 13;
        fname = 'beta';
        d = 5;
    case 5
        lowpass = 48;
        highpass = 30;
        fname = 'gamma';
        d = 4;
end


% Can downsample to improve speed of calculation
% need at least 2*maxfrequency of interest (i.e. Nyquist)
%Valued = resample(Value',1,d)';
fd  =  f %/d; % New sampling frequency
Valued=Value;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency filtering, requires eeglab or other frequency filter.
Vfilt = eegfilt(Valued,fd,highpass,lowpass,0,0,0,'fir1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Vfilt = Vfilt';
% number of time points and Regions of Interest
[m,R] = size(Vfilt);  


% cuts edge points from hilbert transform
cut = 10;


%%  Multivariate leakage correction over whole time course

Timed = linspace(Time(1),Time(end),m);
Vlk =symmetric_orthogonalise(Vfilt,true);


%%  Plots of data before/after multivariate leakage correction
plot(Timed,Vfilt(:,1)); hold on; plot(Timed,Vlk(:,1),'r')

node = 50;

figure(1); clf;
subplot 221; imagesc(cov(Vfilt)); colorbar;
title('Covariance of non-corrected data')

subplot 222; imagesc(cov(Vlk)); colorbar;
title('Covariance of multivariate corrected data')

subplot 223; [Pxx,F] = pwelch(Vfilt(:,node),[],[],1024,fd);
plot(F,Pxx); hold on;
[Plk,F] = pwelch(Vlk(:,node),[],[],1024,fd);
plot(F,Plk,'r'); xlim([highpass-2, lowpass+2]);
xlabel('Hz'); title('Power Spectral Density after frequency filtering');
legend('non-corrected data','multivariate correction')

subplot 224;
plot(Timed,Vfilt(:,node));
hold on
plot(Timed,Vlk(:,node),'r');
legend('non-corrected data','multivariate correction')
xlabel('Time (s)')
title('Example time course after frequency filtering')

set(gcf,'Name','Comparison of data before/after multivariate leakage correction','color','w')
%% Amplitude envelope correlation

ht = hilbert(Vlk);
ht = ht(cut+1:end-cut,:);
env = abs(ht);

%%%%%%%%%%%%%%%

AEC1 = corr(env);
AEC1 = AEC1.*~eye(R);
figure(2); clf; subplot 221
imagesc(AEC1); colorbar;
set(gca,'YTick',1:82,'YTickLabel',LABELS);
title('AEC, multivariate correction')

%% Connectivity estimated with sliding windows

% Bandwidth
B = lowpass-highpass;
% Length of window
T = 10;              % in seconds
N = round(T*fd/2)*2; % in data points, needs to be multiple of 2 for 50% overlap

% Number of windows, with 50% overlap
K = fix((m-N/2)/(N/2));

aec = zeros(R,R,K);
aecp = zeros(R,R,K);

% loop over time windows
for k = 1:K
    
    ibeg = (N/2)*(k-1) + 1;
    iwind = ibeg:ibeg+N-1;
    
    %% Multivariate leakage correction in window for AEC
    
    Vlk = symmetric_orthogonalise(Vfilt(iwind,:),true);
    ht = hilbert(Vlk);
    ht = ht(cut+1:end-cut,:);
    ht = bsxfun(@minus,ht,mean(ht,1));
    % Envelope
    env = abs(ht);
    aec(:,:,k) = corr(env);
    
    
    %% Pairwise leakage correction in window for AEC
    
    % Loops around all possible ROI pairs
    for jj = 1:R
        y = Vfilt(iwind,jj);
        ii =  [1:jj-1,jj+1:R];
        for iii =  1:R-1
            x = Vfilt(iwind,ii(iii));
            % Orthogonalise x with respect to y
           % xc = leakage_reduction(x,y);
            
            % If not using linux, leakage_reduction is equivalent to:
             beta_leak = pinv(y)*x;
             xc = x - y*beta_leak;            
                       
            ht = hilbert([xc,y]);
            ht = ht(cut+1:end-cut,:);
            ht = bsxfun(@minus,ht,mean(ht,1));
            % Envelope
            env = abs(ht);
            c = corr(env);
            aecp(ii(iii),jj,k) = c(1,2);
        end
    end
    
    %%%%%%%%%%%%%%%
    
    fprintf('Calculated AEC for window %d/%d\n',k,K)
end

% Average amplitude correlations over all windows with multivariate
% correction
AEC = mean(aec,3);
AEC = AEC.*~eye(R);  % Set diagonal elements to zero.

figure(2); subplot 222
imagesc(AEC);colorbar
set(gca,'YTick',1:82,'YTickLabel',LABELS);
title(sprintf('AEC, multivariate correction %.1fs window average',T))


% Average amplitude correlations over all windows with pairwise
% correction. Correction is asymmetric so we take the average of the
% elements above and below the diagonal:
% e.g. ( corr(env(1)', env(2)) +  corr(env(1),env(2)') )/2,
% where (1) is an ROI and env' indicates a corrected envelope.
aecp = (aecp + permute(aecp,[2,1,3]))/2;
AECp = mean(aecp,3);
figure(2); subplot 121
imagesc(AECp);colorbar
set(gca,'YTick',1:82,'YTickLabel',LABELS);
set(gca,'XTick',1:82,'XTickLabel',LABELS);
title(sprintf('AEC, pairwise correction %.1fs window average',T))


%% No correction + PLI calculation
ht = hilbert(Vfilt);
ht = ht(cut+1:end-cut,:);
ht = bsxfun(@minus,ht,mean(ht,1));
% Phase information
theta = angle(ht);

% Bandwidth
B = lowpass-highpass;
% Window duration for PLI calculation
T = 100/(2*B);                % ~100 effective points
N = round(T*fd/2)*2;
K = fix((m-N/2-cut*2)/(N/2)); % number of windows, 50% overlap
V = nchoosek(R,2);            % number of ROI pairs
pli = zeros(V,K);

% Loop over time windows
for k = 1:K
    
    ibeg = (N/2)*(k-1) + 1;
    iwind = ibeg:ibeg+N-1;
    
    % loop over all possible ROI pairs
    for jj = 2:R
        ii = 1:jj-1;
        indv = ii + sum(1:jj-2);
        % Phase difference
        RP = bsxfun(@minus,theta(iwind,jj),theta(iwind, ii));
        srp = sin(RP);
        pli(indv,k) = abs(sum(sign(srp),1))/N;
        
    end
end

% Convert to a square matrix
ind = logical(triu(ones(R),1));
PLI = zeros(R);
% Average over windows
PLI(ind) = mean(pli,2);
PLI = PLI + PLI';

figure(2); subplot 122
imagesc(PLI);colorbar; caxis([min(PLI(~eye(R))), max(PLI(~eye(R)))])
set(gca,'YTick',1:82,'YTickLabel',LABELS);
set(gca,'XTick',1:82,'XTickLabel',LABELS);
title(sprintf('PLI, with %.1fs windows',T))

set(gcf,'Name','Connectivity matrices','color','w')