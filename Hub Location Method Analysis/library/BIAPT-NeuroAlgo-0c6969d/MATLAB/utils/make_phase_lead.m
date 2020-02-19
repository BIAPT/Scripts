function [phase_lead_matrix] = make_phase_lead(dpli)
    size_m = length(dpli);
    phase_lead_matrix = zeros(size_m,size_m);

    %Here we make a matrix of phase lead bounded from 0 to 1
    for i = 1:size_m
        for j = 1:size_m
            if(i == j)
                phase_lead_matrix(i,j) = 0;
            elseif(dpli(i,j) <= 0.5)
               phase_lead_matrix(i,j) = 0; 
            else
                phase_lead_matrix(i,j) = (dpli(i,j) - 0.5)*2;
            end
        end
    end
   
end
