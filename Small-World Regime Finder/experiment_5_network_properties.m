% Danielle Nadin 2020-10-11

% Modified from previous experiments.

% Compute wPLI using NeuroAlgo, binarize according to chosen threshold,
% compute binary random networks and compute normalized network properties.
% Allows for processing of multiple participants, time points and frequency
% bands simultaneously.

% wPLI Properties
window_size = 10; % This is in seconds and will be how we chunk the whole dataset
number_surrogate = 20; % Number of surrogate wPLI to create
p_value = 0.05; % the p value to make our test on
step_size = window_size;

% Network Properties Parameters
current_threshold = 0.1;
number_random_network = 10;

for subject = 1:9
    switch subject
        case 1
            sname = 'MDFA03';
        case 2
            sname = 'MDFA05';
        case 3
            sname = 'MDFA06';
        case 4
            sname = 'MDFA07';
        case 5
            sname = 'MDFA10';
        case 6
            sname = 'MDFA11';
        case 7
            sname = 'MDFA12';
        case 8
            sname = 'MDFA15';
        case 9
            sname = 'MDFA17';
    end
    
    for bp = 3
        switch bp
            case 1
                bpname = ' delta';
                frequency_band = [1 4];
            case 2
                bpname = ' theta';
                frequency_band = [4 8];
            case 3
                bpname = ' alpha';
                frequency_band = [8 13];
            case 4
                bpname = ' beta';
                frequency_band = [13 30];
        end
        
        %Create output variable for each participant/frequency band. Each
        %row will be the result for a specific state.
        binary_clustering = zeros(8,1); % 8 is number of states
        binary_geff = zeros(8,1);  
        bsw = zeros(8,1);
        modularity = zeros(8,1);
            
        for state = 1:8
            switch state
                case 1  
                    statename = ' eyes closed 1';
                case 2
                    statename = ' induction first 5 min';
                case 3
                    statename = ' emergence first 5 min';
                case 4
                    statename = '_EML30';
                case 5
                    statename = '_EML10';
                case 6
                    statename = ' emergence last 5 min';
                case 7
                    statename = ' eyes closed 3';
                case 8
                    statename = ' eyes closed 8';
            end
            
            % Load data
            filename = [sname statename '.set'];
            filepath = ['D:\Motif analysis\MDFA\Cleaned data (.set)\' sname];
            recording = load_set(filename, filepath);
            
            % Compute wPLI
            result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
            channels_location = result_wpli.metadata.channels_location;
            pli_matrix = result_wpli.data.avg_wpli;
            pli_matrix = filter_non_scalp(pli_matrix, channels_location);
            
            % Thresholding and binarization using the chosen threshold
            t_network = threshold_matrix(pli_matrix, current_threshold);
            b_network = binarize_matrix(t_network);
            
            % Find average path length
            [b_lambda,b_geff,~,~,~] = charpath(distance_bin(b_network),0,0);
            
            % Find clustering coefficient
            clustering_coef = clustering_coef_bu(b_network);
            
            % Find modularity
            [M,mod] = community_louvain(b_network,1); %community, modularity
            
            % Calculate the null network parameters
            random_networks = zeros(number_random_network,length(pli_matrix),length(pli_matrix));
            parfor r = 1:number_random_network
                disp(strcat("Random network #",string(r)));
                [random_networks(r,:,:),~] = randmio_und(b_network,10);    % generate random matrix
            end
            
            % Find properties for `number_random_network` random network
            total_random_geff = 0;
            total_random_clustering_coef = 0;
            for r = 1:numb er_random_network
                % Create the random network based on the pli matrix instead of the
                % binary network
                random_b_network = squeeze(random_networks(r,:,:));
                
                [rlambda,rgeff,~,~,~] = charpath(distance_bin(random_b_network),0,0);   % charpath for random network
                random_clustering_coef = clustering_coef_bu(random_b_network); % cc for random network
                
                total_random_geff = total_random_geff + rgeff;
                total_random_clustering_coef = total_random_clustering_coef + nanmean(random_clustering_coef);
                
            end
            
            rgeff = total_random_geff/number_random_network;
            global_random_clustering_coef = total_random_clustering_coef/number_random_network;
            
            % Normalize network properties against random network
            binary_clustering(state,1) = nanmean(clustering_coef) / global_random_clustering_coef; % normalized clustering coefficient
            binary_geff(state,1) = b_geff / rgeff;  % global efficiency
            bsw(state,1) = binary_clustering(state,1)*binary_geff(state,1);
            modularity(state,1) = mod; % Note: modularity doesn't need to be normalized against random networks
        end
        
        %When you've looped through all the states for a given
        %participant/freq band, print the result to excel file.
        dlmwrite(['D:\Motif analysis\MDFA\Network analysis\' sname bpname '_cc.csv'], binary_clustering);
        dlmwrite(['D:\Motif analysis\MDFA\Network analysis\' sname bpname '_geff.csv'], binary_geff);
        dlmwrite(['D:\Motif analysis\MDFA\Network analysis\' sname bpname '_bsw.csv'], bsw);
        dlmwrite(['D:\Motif analysis\MDFA\Network analysis\' sname bpname '_mod.csv'], modularity);
    end
end