function [avg] = average_connectivity(matrix)
    % AVERAGE CONNECTIVITY: Given a matri will calculate the average
    % connectivity (globally)
    avg = mean(squeeze(mean(matrix,2)),2)';
end
