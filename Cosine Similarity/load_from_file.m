function [data] = load_from_file(title)
%LOAD_FROM_FILE helper function to load data from file
%   title is used to show a title in the file explorer
    [file,path,~] = uigetfile('*.mat',title);
    full_path = strcat(path,file);
    
    if(file ~= 0)
        data = load(full_path);
    else
        error("You have to select a file.")
    end
end

