%% In this experiment we will test our implementation of morans_I
% If we have a checkered board it means -1, if we have an half 1 half 0
% board we have + 1, if we have a random board we have 0.

grid = zeros(5,5);
counter = 1;
for i = 1:5
   for j = 1:5
       if(mod(counter,2) == 0)
           grid(i,j) = 1;
       end
       counter = counter + 1;
   end
end


weight_matrix = create_square_weight_matrix(5);

I = morans_I(grid,weight_matrix);
C = gearys_C(grid,weight_matrix);