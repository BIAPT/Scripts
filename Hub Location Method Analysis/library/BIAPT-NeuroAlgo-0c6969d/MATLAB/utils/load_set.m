function [recording] = load_set(file_name,path)
%LOAD_EEG will load the EEG data
%   file_name: name of the file to load
%   path: path to that file
%
%   recording: instance of a Recording containing the eeg data
    
    % Currently supported format: .set files
    data = pop_loadset(file_name,path);
    recording = Recording(data.data, data.srate, data.nbchan, data.chanlocs);
end

