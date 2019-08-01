%% Spatial Weights Matrix Calculation
% Here we will use euclidian spatial decay to create our weight matrix
% To start we will calcualte it on one eeg location file

%% Load the data
data = load('EEG_info_WSAS15.mat');
channel_location = data.EEG_info.chanlocs;

%% Init the weight matrix
num_channels = length(channel_location);
weight_matrix = zeros(num_channels,num_channels);

%% Populating the Weights matrix
for i = 1:num_channels
    first_point = [channel_location(i).X, channel_location(i).Y, channel_location(i).Z];
    for j = 1:num_channels
        second_point = [channel_location(j).X, channel_location(j).Y, channel_location(j).Z];
        weight_matrix(i,j) = euclidean_distance(first_point,second_point);
    end
end

%% Normalization with min max
weight_matrix = min_max_normalization(weight_matrix); % This give use normalized distance



%% Helper functions
function distance = euclidean_distance(first_point, second_point)
    distance = sqrt(sum((first_point - second_point) .^ 2));
end

function normalized_matrix = min_max_normalization(matrix)
    normalized_matrix = (matrix - min(matrix)) ./ (max(matrix) - min(matrix)); % This gives us result bounded by [0 and 1]
    normalized_matrix = abs(1 - normalized_matrix); % This give us normalized proximity
    
    % Restore the symmetry and make sure the row and column sum to 1
    for i = 1:length(matrix)
       total = sum(normalized_matrix(i,:));
       for j = 1:length(matrix)
          normalized_matrix(i,j) = normalized_matrix(i,j)/total;
          normalized_matrix(j,i) = normalized_matrix(i,j);
       end
    end
end