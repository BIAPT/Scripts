%{
    Yacine Mahdid 2019-11-01
    This is code taken from the AEC vs wPLI scripts repository, stripped
    down for what is really needed and augmented with the graph theory
    features

    -> We will be generating both the AEC and the wPLI
    -> We will also NOT be doing the normalization here (so we can
    experiment with normalization on the machine learning part of the
    project)
    -> We will keep all the target epochs, there is no need to limit
    ourselves here
    -> In the same vein we will extract all the frequency and not codify
    the labels
    -> We will also write directly to a CSV file like we did for the IEEE
    challenge

%}

%% Variable Initialization
% Data file paths
data_path = "/home/yacine/Documents/AEC vs PLI/data/";

% Experiment Parameters
frequencies = {"alpha","beta","delta","theta","gamma"};
epochs = {'ec1','emf5','ec1','if5','emf5','eml5','ec3','ec8'};
% Graph theory paramters
num_regions = 82; % Number of source localized regions
num_null_network = 10; % Number of null network to create 
bin_swaps = 10;  % used to create the null network
weight_frequency = 0.1; % used to create the null network
t_level = 0.1; % Threshold level (keep 10%)

%% Write the header of the CSV file

%% Write the body of the CSV file containing the data
% We iterate over all the possible permutation and create our filename to
% load
for f_i = 1:length(frequencies)
   for e_i = 1:length(epochs)
       % Get our variables
       frequency = frequencies(f_i);
       epoch = epochs(e_i);

       % Here we process one file and we need to create the filename
       % Need to process both aec and pli at the same time to equalize them
       aec_filename = strcat(data_path,"aec_",epoch,"_",frequency,".mat");
       pli_filename = strcat(data_path,"pli_",epoch,"_",frequency,".mat");
       
       % we load it
       aec_data = load(aec_filename);
       aec_data = aec_data.AEC_OUT;
       pli_data = load(pli_filename);
       pli_data = pli_data.PLI_OUT;
       
       num_participant = length(aec_data);
       
       % Iterate on each participants
       for p_i = 1:number_participant
            % fix aec reverse orientation compared to pli
            aec_data{p_i} = permute(aec_data{p_i},[3 2 1]);
            
            % match the size of the two datasets
            pli_window_length = size(pli_data{p_i},1);
            aec_window_length = size(aec_data{p_i},1);

            min_window_length = min([pli_window_length aec_window_length]);
            pli_data{p_i} = pli_data{p_i}(1:min_window_length,:,:);
            aec_data{p_i} = aec_data{p_i}(1:min_window_length,:,:);
            
            % calculate the feature for both aec and pli
            for w_i = 1:min_window_length
                aec_graph = squeeze(aec_data{p_i}(w_i,:,:));
                pli_graph = squeeze(pli_data{p_i}(w_i,:,:));
                
                X_aec = generate_graph_feature_vector(aec_data, num_null_network, bin_swaps, weight_frequency, t_level);
                X_pli = generate_graph_feature_vector(pli_data, num_null_network, bin_swaps, weight_frequency, t_level);
                
                % Write both of them into the csv file
            end
       end
    end
end