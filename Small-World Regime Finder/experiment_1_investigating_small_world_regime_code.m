%Danielle Nadin 11-12-2019
%Sweep through range of network thresholds and compute binary small-worldness to determine 
% the 'small-world regime range' as defined in Basset et al (2008). 

% modified by Yacine Mahdid 2019-12-12
clear;
% Experiment Variables
filename = 'MDFA17_BASELINE.set';
filepath = 'C:\Users\biapt\Desktop\motif fix\mdfa17_data';


%details on data file
sname = 'MDFA17'; %participant ID
statename = ' BASELINE'; %epoch
sample_frequency = 500;
total_length = 266.160; %length of epoch
EEG_chan = [2,3,4,5,6,7,9,10,11,12,13,15,16,18,19,20,22,23,24,26,27,28,29,30,31,33,34,35,36,37,39,40,41,42,45,46,47,50,51,52,53,54,55,57,58,59,60,61,62,64,65,66,67,69,70,71,72,74,75,76,77,78,79,80,82,83,84,85,86,87,89,90,91,92,93,95,96,97,98,100,101,102,103,104,105,106,108,109,110,111,112,115,116,117,118,122,123,124,129]; %channels in the data file

%sliding window length
win = 10; % in seconds
win = win*sample_frequency; % in pts
number_random_network = 50;

%range of thresholds to sweep
threshold_range = 0.01:0.10:0.61;
%bandpass = [1,30; 1,4; 4,8; 8,13; 13,30];  
%bandpass_name = {'all', 'delta', 'theta', 'alpha', 'beta'};

bandpass = [8,13];
bandpass_name = {'alpha'};
for bp = 1:length(bandpass)
    
    % Get the right bandpass
    bpname = bandpass_name(bp);
    lp = bandpass(bp,1);
    hp = bandpass(bp,2);
    
    %import data
    eeg = pop_loadset(filename, filepath);
    [dataset, ~, ~] = pop_eegfiltnew(eeg, lp, hp);
    filt_data = dataset.data';
    
    PLI = w_PhaseLagIndex(filt_data);

    A = sort(PLI);
    B = sort(A(:));
    X = B(1:length(B)-length(EEG_chan)); % Remove the 1.0 values from B (correlation of channels to themselves)


    all_b_mat = zeros(length(threshold_range),length(PLI),length(PLI));
    for j = 1:length(threshold_range) %loop through thresholds
        disp(strcat("Did : ", string(j/length(threshold_range))));
        current_threshold = threshold_range(j);
        index = floor(length(X)*(1-current_threshold)); %define cut-off for top % of connections
        threshold = X(index); % Values below which the graph will be assigned 0, above which, graph will be assigned 1

        %Binarise the PLI matrix
        b_mat = binarize(PLI, threshold);
        all_b_mat(j,:,:) = b_mat;

        % Find average path length
        [b_lambda,geff,~,~,~] = charpath(distance_bin(b_mat),0,0);   % binary charpath

        % Find clustering coefficient
        clustering_coef = clustering_coef_bu(b_mat);

        % Find properties for random network
        total_random_lambda = 0;
        total_random_clustering_coef = 0;
        for r = 1:number_random_network
            disp(strcat("Random network #",string(r)));
            [random_b_mat,~] = null_model_und_sign(b_mat,10,0.1);    % generate random matrix
            [rlambda,rgeff,~,~,~] = charpath(distance_bin(random_b_mat),0,0);   % charpath for random network
            random_clustering_coef = clustering_coef_bu(random_b_mat); % cc for random network

            total_random_lambda = total_random_lambda + rlambda;
            total_random_clustering_coef = total_random_clustering_coef + nanmean(random_clustering_coef);
        end

        rlambda = total_random_lambda/number_random_network;
        global_random_clustering_coef = total_random_clustering_coef/number_random_network;


        binary_clustering(j) = nanmean(clustering_coef) / global_random_clustering_coef; % normalized clustering coefficient
        binary_charpath(j) = b_lambda / rlambda;  % normalized path length

        binary_small_worldness = binary_clustering(j,i)/binary_charpath(j,i); % normalized binary smallworldness
        bsw(j,i) = binary_small_worldness; 
    end
    
    figure
    plot(threshold_range,mean(bsw,2))
    title([sname ' ' statename ' ' bp ' ' threshold_range(1) ' to ' threshold_range(end)])
    xlabel('Network threshold (%)')
    ylabel('Binary small-worldness')
    
end

% helper function to binarize the matrix
function [b_data] = binarize(data, threshold_value)
    num_channel = length(data);
    b_data = zeros(num_channel, num_channel);
            
    for m = 1:num_channel
        for n = 1:num_channel
            if (m ~= n) && (data(m,n) > threshold_value)
                b_data(m,n) = 1;
            end
        end
    end
end