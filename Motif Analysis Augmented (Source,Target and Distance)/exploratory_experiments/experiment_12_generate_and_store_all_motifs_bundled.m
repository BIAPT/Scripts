
%% Variables Initalization
participants = {'MDFA03','MDFA05','MDFA06','MDFA07','MDFA11','MDFA12','MDFA15'};
states = {'EC1', 'IF5', 'EF5','EL30','EL10','EL5','EC8'};


% Experiment variables
% for Motif Analysis
number_rand_network = 1;
bin_swaps = 10;
weight_frequency = 0.1;
frequency = "alpha";


%% Iterating over the files
for p = 1:length(participants)
    participant = participants{p};
    input_path = strcat('C:/Users/biapt/Desktop/motif_analysis_dst/dPLI/',participant);
    output_path = strcat('E:/research projects/motif_analysis/motif/',participant,'.mat');
    state_data = [];
    for e_i = 1:length(states)
        % Load the data
        disp(strcat("Motif analysis on dpli value from ", participant," at ", states{e_i}));

        filename = strcat(input_path,filesep,states{e_i},filesep,'dpli_',frequency,'.mat');
        data = load(filename);
        dpli_matrix = data.z_score;

        filename = strcat(input_path,filesep,'eeg_info.mat');
        data = load(filename);
        channels_location = data.EEG_info.chanlocs;

        % make a phase lead matrix using the average dPLI
        network = make_phase_lead(dpli_matrix);

        % Filter the channels location to match the filtered motifs
        [network,channels_location] = filter_non_scalp(network,channels_location);

        % Calculate the frequency/source/target/distance of each motifs
        motifs = struct();
        [motifs.frequency, motifs.source, motifs.target, motifs.distance] = motif_3(network, channels_location, number_rand_network, bin_swaps, weight_frequency);

        state_data = [state_data, motifs];
    end

    %% Save the data
    save(output_path, 'state_data');
end

% Helper function to filter channels in a matrix
function  [matrix,channels_location] = filter_non_scalp(matrix,channels_location)
%FILTER_NON_SCALP Summary of this function goes here
%   Detailed explanation goes here
    non_scalp_channel_label = {'E127', 'E126', 'E17', 'E128', 'E125', 'E21', 'E25', 'E32', 'E38', 'E44', 'E14', 'E8', 'E1', 'E121', 'E114', 'E43', 'E49', 'E56', 'E63', 'E68', 'E73', 'E81', 'E120', 'E113', 'E107', 'E99', 'E94', 'E88', 'E48', 'E119'};

    for i=1:length(non_scalp_channel_label)
        current_label = non_scalp_channel_label{i};
        for j=1:length(channels_location)
           if(strcmp(channels_location(j).labels,current_label))
               channels_location(j) = [];
               matrix(j,:) = [];
               matrix(:,j) = [];
               break;
           end
        end
    end
end