function [configuration] = get_configuration()
%GET_CONFIGURATION Getter for the configuration data
%   This will parse the configuration.txt file to get the right variables
%   out

    %% Variables Intialization
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    configuration_path = strcat(filepath,'/../configuration.txt');
    file_id = fopen(configuration_path);
    
    configuration.is_verbose = get_is_verbose(file_id);
    %configuration.saving_directory = get_saving_directory(file_id);
    configuration.bandpass = get_bandpass(file_id);
    
    fclose(file_id);
end

function [is_verbose] = get_is_verbose(file_id)
    is_verbose = 0;
    line = fgetl(file_id);
    while(line ~= -1)
        line = strtrim(line);
        line_data = split(line,"=");
        
        if(length(line_data) > 1)
            identifier = strtrim(line_data(1));
            value = strtrim(line_data(2));
            if(strcmp(identifier,"is_verbose"))
                is_verbose = str2double(value{1});
                break;
            end
        end
        line = fgetl(file_id);
    end
    frewind(file_id);
end

function [bandpass] = get_bandpass(file_id)
     
    %% Variable Initialization
    bandpass = struct();
    bandpass.alpha = [];
    bandpass.beta = [];
    bandpass.theta = [];
    bandpass.gamma = [];
    bandpass.delta = [];
    
    line = fgetl(file_id);
    while(line ~= -1)
        line = strtrim(line);
        line_data = split(line,"=");
        
        if(length(line_data) > 1)
            identifier = strtrim(line_data(1));
            value = strtrim(line_data(2));
            if(strcmp(identifier,"alpha"))
                bandpass.alpha = str2num(value{1});
            elseif(strcmp(identifier,"beta"))
                bandpass.beta = str2num(value{1});
            elseif(strcmp(identifier,"delta"))
                bandpass.delta = str2num(value{1});
            elseif(strcmp(identifier,"gamma"))
                bandpass.gamma = str2num(value{1});
            elseif(strcmp(identifier,"theta"))
                bandpass.theta = str2num(value{1});
            end
        end
        line = fgetl(file_id);
    end
    frewind(file_id);
end

