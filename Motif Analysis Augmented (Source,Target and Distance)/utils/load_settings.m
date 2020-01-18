function [settings] = load_settings()
%   LOAD_SETTINGS will load the settings from settings.txt
%   this function read the settings.txt file and transform it into a 
%   structure that can be used to access global parameters.
%   the settings structure will contain the input and output path
%   for these experiments.

    %% Open settings.txt file
    [filepath,~,~] = fileparts(mfilename('fullpath'));
    file_id = fopen(strcat(filepath,'/../settings.txt'));
    
    %% Read settings.txt and populate the settings structure
    settings = struct();
    
    % Populate the input path
    input_path_line = fgetl(file_id);
    input_path_data = split(strtrim(input_path_line),"=");
    settings.input_path = strtrim(input_path_data{2});
    
    % Populate the output path
    output_path_line = fgetl(file_id);
    output_path_data = split(strtrim(output_path_line),"=");
    settings.output_path = strtrim(output_path_data{2});
     
    fclose(file_id);
end