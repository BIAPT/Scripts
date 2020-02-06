
[filepath,name,ext] = fileparts(mfilename('fullpath'));
test_data_path = strcat(filepath,'data/MDFA05_Cleaned');
recording = load_set('MDFA05_eyes_closed_1_brainonly.set',test_data_path);

% OR load a saved text file
data = readtable('test_electrode.txt');
data = table2array(data);

plot_circle(data,recording,0.05, 'Midline');
plot_circle(data,recording,0.05, 'Inter');
plot_circle(data,recording,0.05, 'Intra');

test=result_wpli.data.avg_wpli;
