function find_network_properties3

%   This function defines a network like Joon's paper, but normalizes
%   properties according to random networks

samp_freq = 250;
network_thresh = 0.05; %vary to see how stable our results are
win = 10;   % number of seconds of EEG window
% total_length = 300;    % total number of seconds of EEG epoch


for subject = 5
    switch subject
        case 1
            sname = 'WSAS05';
            EEG_chan = [2    3    4    5    6    7    9   10   11   12   13   15   16   18   19   20   22   23   24   26   27   28   29   30   31   33   34   35   37   39   40   41   42   45   46   47   51   52   53   55   58   59   60   61   64   65   66   67   69   74   75   77   78   79   80   82   83   84   85   86   87   89   90   91   92   93   95   96   97   98  101  102  103  105  106  108  109  110  111  112  116  117  118  122  123  124  129];
            total_length = 300;
        case 2
            sname = 'WSAS07';
            EEG_chan = [2    3    4    5    6    7    9   10   11   12   13   15   16   18   19   20   22   23   24   26   27   28   29   30   33   34   35   36   39   40   41   45   46   47   50   51   52   53   54   58   59   60   61   62   64   65   66   67   69   70   71   72   74   75   76   77   78   79   82   83   84   85   86   87   89   90   91   92   93   95   96   97   98  101  102  103  104  105  108  109  110  111  112  115  116  117  118  122  123  124  129];
            total_length = 300;
        case 3
            sname = 'WSAS13';
            EEG_chan = [2    3    4    5    9   10   11   12   15   16   18   19   20   22   23   24   26   27   28   29   30   31   33   34   35   36   37   39   40   41   42   45   46   47   50   51   52   53   54   58   59   60   61   62   64   65   66   67   69   70   71   72   74   75   76   77   78   79   80   82   84   85   86   87   90   91   92   93   95   96   97   98  101  102  103  104  105  106  108  109  110  111  112  115  116  117  118  122  123  129];
            total_length = 300;
        case 4
            sname = 'WSAS02';
            EEG_chan = [1   2   3   5   6   7   8   9  10  13  14  15  17  18  19  20  21  22  23  25  26  27  29  30  32  33  34  35  36  38  39  40  41  42  43  44  45  46  48  49  50  51  52  53  54  55  57  58  59  60  62  63];
            total_length = 300;
        case 5
            sname = 'WSAS09';
            EEG_chan = [2    3    4    5    6    9   10   11   12   13   15   16   18   19   20   22   23   24   26   29   30   31   33   34   37   39   40   42   45   46   47   52   53   58   59   60   61   62   66   67   69   70   71   72   74   75   76   77   78   79   82   83   84   85   86   89   90   91   92   93   95   96   97   98  101  102  103  105  108  109  110  111  115  116  117  122  123  124  129];
            total_length = 300;
%         case 4
%             sname = 'MDFA07';
%             EEG_chan = [2,3,4,5,6,7,9,10,11,12,13,15,16,17,18,20,21,23,24,25,26,27,28,30,31,32,33,34,36,37,38,40,41,42,45,46,47,48,49,50,52,53,54,55,56,59,60,61,63,65,66,69,70,71,72,73,74,77,78,79,80,81,84,85,86,87,90,91,92,94,95,96,97,98,99,100,101,102,103,104,108,109,110,114,120];
%         case 5
%             sname = 'MDFA10';
%             EEG_chan = [2,3,4,5,6,8,9,10,12,13,15,16,17,19,20,21,23,24,25,26,27,28,30,31,32,33,34,36,37,38,39,42,43,46,47,48,49,50,51,54,54,55,56,57,60,61,62,65,68,69,70,71,72,73,76,77,78,79,80,83,84,85,86,89,90,91,94,95,96,97,98,100,101,102,106,107,108,112,113,118];
%         case 6
%             sname = 'MDFA11';
%             EEG_chan = [2,3,4,5,6,7,9,10,11,12,13,15,16,18,19,20,22,23,24,26,27,28,29,30,31,33,34,35,36,37,39,40,41,42,45,48,49,50,51,52,55,56,57,58,59,62,63,64,67,68,69,72,73,74,75,76,77,80,81,82,83,84,87,88,89,90,93,94,95,98,99,100,101,102,103,105,106,107,108,109,113,114,115,119,120,126];
%         case 7
%             sname = 'MDFA12';
%             EEG_chan = [2,3,4,5,6,7,9,10,11,12,13,15,16,18,19,20,22,23,24,26,27,28,29,30,31,33,34,35,36,37,39,40,41,42,45,46,47,50,51,52,53,54,55,58,59,60,61,62,65,66,67,70,71,72,75,76,77,78,79,80,83,84,85,86,87,90,91,92,93,96,97,98,101,102,103,104,105,106,108,109,110,111,112,116,117,118,122,123,129];
%        case 8 
%            sname = 'MDFA15';
%            EEG_chan = [2,3,4,5,6,7,9,10,11,12,13,15,16,18,19,20,22,23,24,26,27,28,29,30,31,33,34,35,36,37,39,40,41,42,45,46,47,50,51,52,53,54,55,58,59,60,61,62,65,66,67,70,71,72,75,76,77,78,79,80,83,84,85,86,87,90,91,92,93,96,97,98,100,101,102,103,104,105,107,108,109,110,111,115,116,117,121,122,128];
%        case 9
%            sname = 'MDFA17';
%            EEG_chan = [2,3,4,5,6,7,9,10,11,12,13,15,16,18,19,20,22,23,24,26,27,28,29,30,31,33,34,35,36,37,39,40,41,42,45,46,47,50,51,52,53,54,55,58,59,60,61,62,65,66,67,70,71,72,75,76,77,78,79,80,83,84,85,86,87,90,91,92,93,96,97,98,101,102,103,104,105,106,108,109,110,111,112,116,117,118,122,123,129];
    end
    
%     Larray = zeros(10,total_length/win);
%     Carray = zeros(10,total_length/win);
%     geffarray = zeros(10,total_length/win);
%     bswarray = zeros(10,total_length/win);
%     Qarray = zeros(10,total_length/win);

     Larray = zeros(1,floor(total_length/win)); %path length
     Carray = zeros(1,floor(total_length/win)); %clustering coefficient
     geffarray = zeros(1,floor(total_length/win)); %global efficiency
     bswarray = zeros(1,floor(total_length/win)); %small worldness
     Qarray = zeros(1,floor(total_length/win)); %modularity

    
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
    
        for state = 1:3
            switch state
                case 1
                    statename = '_Pre_5min';
                case 2
                    statename = '_Sed_5min';
                case 3
                    statename = '_Post_5min';
%                 case 4
%                     statename = ' emergence last 5 min';
%                 case 5
%                     statename = ' eyes closed 3';
%                 case 6
%                     statename = ' eyes closed 4';
%                 case 7 
%                     statename = ' eyes closed 5';
%                 case 8
%                     statename = ' eyes closed 6';
%                 case 9
%                     statename = ' eyes closed 7';
%                 case 10
%                     statename = ' eyes closed 8';
%                 case 11
%                     statename = ' unconscious epoch 1';
            end
        
            state
%           EEG = pop_loadset('filename', [sname statename '.set'],'filepath',['F:\McDonnell Foundation study\University of Michigan\Anesthesia\' sname '\Resting state analysis']);
            EEG = pop_loadset('filename', [sname statename '.set'],'filepath','C:\Users\Danielle\OneDrive - McGill University\Research\BIAPT Lab\DOC\Motif paper\WSAS09\DATA\5 min segments\');

                
            [dataset, com, b] = pop_eegfiltnew(EEG, lp, hp);    
            filt_data = dataset.data';
        
            b_charpath = zeros(1,floor(total_length/win));
            b_clustering = zeros(1,floor(total_length/win));
            b_geff = zeros(1,floor(total_length/win));
            bsw = zeros(1,floor(total_length/win));
            Q = zeros(1,floor(total_length/win));
            
            for i = 1:(floor((length(filt_data))/(win*samp_freq)))
                
%                 EEG_seg = filt_data((i-1)*win*samp_freq + 1:i*win*samp_freq, EEG_chan);      % Only take win seconds length from channels that actually have EEG
                EEG_seg = filt_data((i-1)*win*samp_freq + 1:i*win*samp_freq, :);
                
                PLI = w_PhaseLagIndex(EEG_seg); %weighted PLI
                      
                A = sort(PLI);
                B = sort(A(:));
                C = B(1:length(B)-length(EEG_chan)); % Remove the 1.0 values from B (correlation of channels to themselves)
            
                index = floor(length(C)*(1-network_thresh)); %top network_thresh% of data
                thresh = C(index);  % Values below which the graph will be assigned 0, above which, graph will be assigned 1
            
            
            % Create a (undirected, unweighted) network based on top network_thresh% of PLI connections    
            for m = 1:length(PLI)
                for n = 1:length(PLI)
                    if (m == n)
                        b_mat(m,n) = 0;
                    else
                        if (PLI(m,n) > thresh)
                            b_mat(m,n) = 1;
                        else
                            b_mat(m,n) = 0;
                        end
                    end
                end
            end          
%                 save([sname '_' bpname '_b_mat.mat'],'b_mat');
                
                % Find average path length
                               
                D = distance_bin(b_mat);
                [b_lambda,geff,~,~,~] = charpath(D,0,0);   % binary charpath
                [W0,R] = null_model_und_sign(b_mat,10,0.1);    % generate random matrix
                
                  % Find clustering coefficient

                C = clustering_coef_bu(b_mat);  
                
                % Find properties for random network
                
                [rlambda,rgeff,~,~,~] = charpath(distance_bin(W0),0,0);   % charpath for random network
                rC = clustering_coef_bu(W0); % cc for random network
                                  
                b_clustering(i) = nanmean(C)/nanmean(rC); % binary clustering coefficient
                b_charpath(i) = b_lambda/rlambda;  % charpath
                b_geff(i) = geff/rgeff; % global efficiency
                
                bsw(i) = b_clustering/b_charpath; % binary smallworldness
                
                [M,modular] = community_louvain(b_mat,1); % community, modularity
                Q(i) = modular;
                
                clear EEG_seg PLI b_mat M modular b_lambda rlambda geff rgeff
                
            end
            
             Larray(state,:) = b_charpath(1,1:floor(total_length/win)); 
             Carray(state,:) = b_clustering;
             geffarray(state,:) = b_geff;
             bswarray(state,:) = bsw;
             Qarray(state,:) = Q;
             
%             Larray = b_charpath; 
%             Carray = b_clustering;
%             geffarray = b_geff;
%             bswarray = bsw;
%             Qarray = Q;
             
             clear EEG filt_data b_charpath b_clustering b_geff bsw Q
        end
    
        dlmwrite(['C:\Users\Danielle\OneDrive - McGill University\Research\BIAPT Lab\DOC\Motif paper\Results\Network Properties\Output\' sname bpname '_Lnorm.csv'], Larray);
        dlmwrite(['C:\Users\Danielle\OneDrive - McGill University\Research\BIAPT Lab\DOC\Motif paper\Results\Network Properties\Output\' sname bpname '_Cnorm.csv'], Carray);
        dlmwrite(['C:\Users\Danielle\OneDrive - McGill University\Research\BIAPT Lab\DOC\Motif paper\Results\Network Properties\Output\' sname bpname '_geffnorm.csv'], geffarray);
        dlmwrite(['C:\Users\Danielle\OneDrive - McGill University\Research\BIAPT Lab\DOC\Motif paper\Results\Network Properties\Output\' sname bpname  '_bsw.csv'], bswarray);
        dlmwrite(['C:\Users\Danielle\OneDrive - McGill University\Research\BIAPT Lab\DOC\Motif paper\Results\Network Properties\Output\' sname bpname   '_Qnorm.csv'], Qarray);
       
        clear Larray Carray geffarray bswarray Qarray
        
    end
    
end

 %figure; plot(Lnorm)
                
