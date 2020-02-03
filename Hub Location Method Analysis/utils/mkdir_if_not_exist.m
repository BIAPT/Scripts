function  [folder_path] = mkdir_if_not_exist(parent_folder, folder)
%   MKDIR_IF_NOT_EXIST create a directory if it doesn't exist
%   this is a wrapper function that will avoid having warning if the
%   directory already exist
%
%   >> folder_path = mkdir_if_not_exist(parent_folder, folder)
%   parent_folder is the parent folder
%   folder is the new folder that need to be created
%   folder_path is the full folder path

    folder_path = strcat(parent_folder, filesep, folder);
    
    if ~exist(folder_path,'dir')
        mkdir(parent_folder, folder);
    end

end

