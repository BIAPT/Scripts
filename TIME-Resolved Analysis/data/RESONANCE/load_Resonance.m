
cd ('/Users/biapt/Documents/RESONANCE_AUDACE/audace_workshop_26_november_2019/danny_fourth_clowning-737755.4923/data/')

files = dir('*.mat');
fulldata=[];

for i=0:length(files)-1
    a=load(string(i)+'.mat');
    new=a.data;
    fulldata=[fulldata,new];
end

save ('/Users/biapt/Documents/Charlotte_Analysis/RESONANCE/fulldata_session4.mat','fulldata')

disp('data saved')