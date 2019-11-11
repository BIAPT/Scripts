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
data_path = "/home/yacine/Documents/aec_vs_wpli/data/";
output_path = "/home/yacine/Documents/aec_vs_wpli/norm_wei_data.csv";

% Experiment Parameters
frequencies = {'alpha'}; %{'alpha','beta','delta','theta'};
epochs = {'ec1','emf5','eml5'}; %{'ec1','if5','emf5','eml5','ec3'}; % we are not using ec8 here because only alpha has it
% Graph theory paramters
num_regions = 82; % Number of source localized regions
num_null_network = 10; % Number of null network to create 
bin_swaps = 10;  % used to create the null network
weight_frequency = 0.1; % used to create the null network
t_level = 0.1; % Threshold level (keep 10%)
transform = 'log'; % this is used for the weighted_global_efficiency

%% Write the header of the CSV file

header = ["p_id", "frequency", "epoch","graph","window"];
for r_i = 1:num_regions
   mean_header = strcat("mean_",string(r_i));
   header = [header, mean_header];
end

for r_i = 1:num_regions
    std_header = strcat("std_",string(r_i));
    header = [header,std_header];
end

for r_i = 1:num_regions
    clust_coeff = strcat("clust_coeff_ ", string(r_i));
    header = [header,clust_coeff];      
end
header = [header,"norm_avg_clust_coeff","norm_g_eff","community","small_worldness"];

% Overwrite the file
delete(output_path);

% Write header to the features file
fileId = fopen(output_path,'w');
for i = 1:(length(header)-1)
    fprintf(fileId,'%s,',header(i));
end
fprintf(fileId,"%s\n",header(length(header)));
fclose(fileId);

%% Write the body of the CSV file containing the data
% We iterate over all the possible permutation and create our filename to
% load
for f_i = 1:length(frequencies)
    frequency = frequencies(f_i);
    disp(strcat("Frequency: ",frequency));
    for e_i = 1:length(epochs)
       % Get our variables
       epoch = epochs(e_i);
       disp(strcat("Epochs: ",epoch));

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
       for p_i = 1:num_participant
           disp(strcat("Participant id: ", string(p_i)));
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
                disp(strcat("Window : ", string(w_i)));
                aec_graph = squeeze(aec_data{p_i}(w_i,:,:));
                pli_graph = squeeze(pli_data{p_i}(w_i,:,:));
                
                X_aec = generate_weighted_graph_feature_vector(aec_graph, num_null_network, bin_swaps, weight_frequency, transform);
                X_pli = generate_weighted_graph_feature_vector(pli_graph, num_null_network, bin_swaps, weight_frequency, transform);
                
                % Write both of them into the csv file
                dlmwrite(output_path, [p_i, f_i, e_i, 0, w_i, X_aec'], '-append');
                dlmwrite(output_path, [p_i, f_i, e_i, 1, w_i, X_pli'], '-append');
       
            end
       end
    end
end