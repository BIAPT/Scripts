function distance = euclidean_distance(first_point, second_point)
    distance = sqrt(sum((first_point - second_point) .^ 2));
end

