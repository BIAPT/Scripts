%Danielle Nadin 11-12-2019
%Sweep through range of network thresholds and compute binary small-worldness to determine 
% the 'small-world regime range' as defined in Basset et al (2008). 

% BEFORE RUNNING: MODIFY LINES 8-12, 18, 20, 46 and 100

%details on data file
sname = 'MDFA03'; %participant ID
statename = ' EC1'; %epoch
samp_freq = 500;
total_length = 266.160; %length of epoch
EEG_chan = [2,3,4,5,6,7,9,10,11,12,13,15,16,18,19,20,22,23,24,26,27,28,29,30,31,33,34,35,36,37,39,40,41,42,45,46,47,50,51,52,53,54,55,57,58,59,60,61,62,64,65,66,67,69,70,71,72,74,75,76,77,78,79,80,82,83,84,85,86,87,89,90,91,92,93,95,96,97,98,100,101,102,103,104,105,106,108,109,110,111,112,115,116,117,118,122,123,124,129]; %channels in the data file

%sliding window length
win = 10;

%range of thresholds to sweep
range = 0.01:0.02:0.6;

for bp = 4
    
    switch bp
        case 1
            bpname = ' all';
            lp = 1;
            hp = 30;
        case 2
            bpname = ' delta';
            lp = 1;
            hp = 4;
        case 3
            bpname = ' theta';
            lp = 4;
            hp = 8;
        case 4
            bpname = ' alpha';
            lp = 8;
            hp = 13;
        case 5
            bpname = ' beta';
            lp = 13;
            hp = 30;
    end
    
    %import data
    EEG = pop_loadset('filename','MDFA03 eyes closed 1.set','filepath','D:\Motif analysis\MDFA\Raw data (.set)\MDFA03');
    [dataset, com, b] = pop_eegfiltnew(EEG, lp, hp);
    filt_data = dataset.data';
    
    %compute graph properties for each epoch using sliding window
    for i = 1:floor(length(filt_data)/(win*samp_freq))
        
        EEG_seg = filt_data((i-1)*win*samp_freq + 1:i*win*samp_freq,:);
        PLI = w_PhaseLagIndex(EEG_seg);
        
        A = sort(PLI);
        B = sort(A(:));
        X = B(1:length(B)-length(EEG_chan)); % Remove the 1.0 values from B (correlation of channels to themselves)
        
        for j = 1:size(range,2) %loop through thresholds
            
            index = floor(length(X)*(1-range(j))); %define cut-off for top % of connections
            thresh = X(index); % Values below which the graph will be assigned 0, above which, graph will be assigned 1
            
            %Binarise the PLI matrix
            b_mat = zeros(size(1:length(PLI),2),size(1:length(PLI),2));
            
            for m = 1:length(PLI)
                for n = 1:length(PLI)
                    if (m ~= n) && (PLI(m,n) > thresh)
                        b_mat(m,n) = 1;
                    end
                end
            end
            
            
            % Find average path length
            [b_lambda,geff,~,~,~] = charpath(distance_bin(b_mat),0,0);   % binary charpath
            
            % Find clustering coefficient
            C = clustering_coef_bu(b_mat);
            
            % Find properties for random network
            [W0,R] = null_model_und_sign(b_mat,10,0.1);    % generate random matrix
            [rlambda,rgeff,~,~,~] = charpath(distance_bin(W0),0,0);   % charpath for random network
            rC = clustering_coef_bu(W0); % cc for random network
            
            b_clustering(j,i) = nanmean(C)/nanmean(rC); % normalized clustering coefficient
            b_charpath(j,i) = b_lambda/rlambda;  % normalized path length
            bsw(j,i) = b_clustering(j,i)/b_charpath(j,i); % normalized binary smallworldness
            
            clear b_mat b_lambda C W0 rlambda rC
            
        end
        
        clear EEG_seg PLI A B X 
        
    end
    
    dlmwrite(['D:\Motif analysis\' sname bpname statename '_bswnorm.csv'], bsw); %location to save data
    
    figure
    plot(range,mean(bsw,2))
    title([sname ' ' statename ' ' bp ' ' range(1) ' to ' range(end)])
    xlabel('Network threshold (%)')
    ylabel('Binary small-worldness')
    
    clear  b_charpath b_clustering bsw
    
end