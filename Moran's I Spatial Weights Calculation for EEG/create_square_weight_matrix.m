function weight_matrix = create_square_weight_matrix(number_square)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

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
    weight_matrix = square_normalization(weight_matrix); % This give use normalized distance
    
    %% Zero the midline
    for i = 1:length(weight_matrix)
       for j = 1:length(weight_matrix)
           if(i == j)
               weight_matrix(i,j) = 0; 
           end
       end
    end
end

