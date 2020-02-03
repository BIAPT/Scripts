%{
    Yacine Mahdid 2020-02-03
    This script is run to make sure that the project uses the right
    depencies by downloading the right libraries and adding to path the
    right folders.
%}

% Setting up the paths
[folder, name, ext] = fileparts(mfilename('fullpath'));
library_path = strcat(folder,filesep,'library');
utils_path = strcat(folder,filesep,'utils');
temp_path = strcat(folder,filesep,'temp.zip');

% Fetch the NA library (using 0.0.1) and saving it to /library
disp("Fetching the ressources (NA.0.0.1)");
url = 'https://api.github.com/repos/BIAPT/NeuroAlgo/zipball/0.0.1';
outfilename = websave(temp_path, url);
unzip(outfilename,library_path)
delete(temp_path)

% Add that folder plus all subfolders to the path.
disp("Adding ressource to path.")
addpath(genpath(library_path));
addpath(genpath(utils_path));