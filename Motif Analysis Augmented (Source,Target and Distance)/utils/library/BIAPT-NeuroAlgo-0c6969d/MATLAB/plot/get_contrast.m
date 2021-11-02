function new= get_contrast(data)
% this function generates the timestep difference of two connectivity
% matrices
dim=size(data);
new = double.empty(0,dim(2),dim(3));

for i=1:dim(1)-1
    j = i+1;    
    new(i,:,:)=data(j,:,:)-data(i,:,:);        
    if rem(i,100)==0
        disp(string(i/dim(1))+"%")
    end
end

