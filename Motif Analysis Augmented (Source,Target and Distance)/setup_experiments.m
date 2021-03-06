%{
    Yacine Mahdid 2020-01-08
    This script is run to make sure that all experiment have the same
    state variable without having to repeat the code everywhere.
%}

% General Experiment Variables
settings = load_settings();
raw_data_path = settings.raw_data_path;
output_path = settings.output_path;

participants = {'MDFA03','MDFA05','MDFA06','MDFA07','MDFA10','MDFA11','MDFA12','MDFA15','MDFA17'};
% states = {'BASELINE','EMF5','EC3','RECOVERY'};
 states = {' eyes closed 1',' emergence first 5 min',' eyes closed 3',' eyes closed 8'};
% participants = {'WSAS07'};
% states = {'_Pre_5min'};

% wPLI Experiment Variables
wpli_param = struct();
wpli_param.frequency_band = [8 13]; % This is in Hz
wpli_param.window_size = 10; % This is in seconds and will be how we chunk the whole dataset
wpli_param.number_surrogate = 20; % Number of surrogate wPLI to create
wpli_param.p_value = 0.05; % the p value to make our test on
wpli_param.step_size = 10;

% dPLI Experiment Variables
dpli_param = struct();
dpli_param.frequency_band = [8 13]; % This is in Hz
dpli_param.window_size = 10; % This is in seconds and will be how we chunk the whole dataset
dpli_param.number_surrogate = 20; % Number of surrogate wPLI to create
dpli_param.p_value = 0.05; % the p value to make our test on
dpli_param.step_size = 10;

% motif Experiment Variable
motif_param = struct();
motif_param.number_rand_network = 100;
motif_param.bin_swaps = 10;
motif_param.weight_frequency = 0.1;

% power spectrum Experiment Variable
power_param = struct();
power_param.frequency_band = [8 13];
power_param.average = 1; %Do you want to generate the average?

% hubs experiment variables
hubs_param = struct();
hubs_param.threshold = 0.05;
hubs_param.average = 1; %Do you want to generate the average?

% graph theory experiment variables
graph_param = struct();
graph_param.threshold = 0.35;
graph_param.number_surrogate = 10;

% The other parameters are recording dependant and will be dynamically
% generated
