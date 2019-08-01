%% Equidistant matrix
% This code is used to work with a square instead of a EEG head

%% Variable initalization
number_square = 10; % This you can modify
X = 1:number_square;
Y = 1:number_square;
number_channels = number_square*number_square;
weight_matrix = zeros(number_channels,number_channels);
channel_location = [];

for i = 1:number_square
    for j = 1:number_square        
        location = struct();
        location.X = X(i);
        location.Y = Y(j);
        channel_location = [channel_location, location];
    end
end

%% Populating the Weights matrix
for i = 1:number_channels
    first_point = [channel_location(i).X, channel_location(i).Y];
    for j = 1:number_channels
        second_point = [channel_location(j).X, channel_location(j).Y];
        weight_matrix(i,j) = euclidean_distance(first_point,second_point);
    end
end

%% Normalization
weight_matrix = min_max_normalization(weight_matrix); % This give use normalized distance

%% Helper functions
function distance = euclidean_distance(first_point, second_point)
    distance = sqrt(sum((first_point - second_point) .^ 2));
end

function normalized_matrix = min_max_normalization(matrix)
    normalized_matrix = (matrix - min(matrix)) ./ (max(matrix) - min(matrix)); % This gives us result bounded by [0 and 1]
    normalized_matrix = abs(1 - normalized_matrix); % This give us normalized proximity
    
    
    symmetric_normalized_matrix = matrix;
    % Restore the symmetry and make sure the row and column sum to 1
    for i = 1:length(matrix)
       total = sum(normalized_matrix(i,:));
       for j = 1:length(matrix) 
          symmetric_normalized_matrix(i,j) = normalized_matrix(i,j)/total;
          symmetric_normalized_matrix(j,i) = symmetric_normalized_matrix(i,j); 
       end
    end
    
    normalized_matrix = symmetric_normalized_matrix;
end
