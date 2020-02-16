function [output] = reorder_channels(input)
%EEG_MANIPULATION is a wrapper function to call the python function
    
    % Save the original directory and move to the other path
    og_dir = pwd();
    [own_path, ~, ~] = fileparts(mfilename('fullpath'));    
    cd(own_path);
    
    % Call the python function
    py.eeg_manipulation.reorder.reorder_channels(input)
    
    % Go back to the original directory
    cd(og_dir)
end

