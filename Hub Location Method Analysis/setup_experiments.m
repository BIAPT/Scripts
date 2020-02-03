%{
    Yacine Mahdid 2020-01-08
    This script is run to make sure that all experiment have the same
    state variable without having to repeat the code everywhere.
%}

% General Experiment Variables
settings = struc();
settings.raw_data_path = '/home/yacine/Documents/consciousness_data';
settings.ouput_path = '/home/yacine/Documents/hub_location_output';

% Set participant to be one of {'MDFA03','MDFA05','MDFA06','MDFA07','MDFA10','MDFA11','MDFA12','MDFA15','MDFA17'};
settings.participant = 'MDFA03';
% Set state to be one of : {'BASELINE', 'IF5', 'EMF5', 'EML30','EML10','EML5', 'EC3', 'RECOVERY'};
settings.state = 'BASELINE';

% wPLI Experiment Variables
settings.wpli.frequency_band = [8 13]; % This is in Hz
settings.wpli.window_size = 10; % This is in seconds and will be how we chunk the whole dataset
settings.wpli.number_surrogate = 20; % Number of surrogate wPLI to create
settings.wpli.p_value = 0.05; % the p value to make our test on
settings.wpli.step_size = 1;

% Hub Location Variables
% TODO