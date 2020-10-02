%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Adapt excel files                           %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%     Choose following variables: 

part="NET-ICU-002-MG"

missing_el_baseline = ["E006","E011","E005","E016","E112","E105","E010","E004",...
    "E118","E111","E104","E103","E110","E117","E124","E003","E123","E116",...
    "E109","E122","E144", "E126", "E125", "E127", "E095", "E055", "E078", "E079"]
missing_el_inter = ["E006","E011","E005","E016","E112","E105","E010","E004",...
    "E118","E111","E104","E103","E110","E117","E124","E003","E123","E116",...
    "E109","E122","E144", "E126", "E125", "E127", "E055", "E048"]
missing_el_post = ["E006","E011","E005","E016","E112","E105","E010","E004",...
    "E118","E111","E104","E103","E110","E117","E124","E003","E123","E116",...
    "E109","E122","E144", "E126", "E125", "E127", "E055", "E048"]


%%%%%%%%%%%%%%%%%%%%%%   START         %%%%%%%%%%%%%%%%%%%%
folder = 'C:\Users\BIAPT\Desktop\'+part
cd(folder)

expath = "C:\Users\BIAPT\Desktop\Material\Electrode number and region.xlsx"
[~,~,el_left]=xlsread(expath,"LEFT");
[~,~,el_right]=xlsread(expath,"RIGHT");

%%%% BASELINE
for i = 1:length(missing_el_baseline)
    match=length(find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_baseline(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_baseline(i))))))
        el_left(item,:)=[]
    end
    
    match=length(find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_baseline(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_baseline(i))))))
        el_right(item,:)=[]
    end
end

el_left=table(el_left,'VariableNames',{'Left'});
el_right=table(el_right,'VariableNames',{'Right'});
filename =folder+'\' +part+'_electrodes_sedon1.xlsx';
writetable(el_left,filename);
writetable(el_right,filename,'sheet',2)

%%%% Interruption
[~,~,el_left]=xlsread(expath,"LEFT");
[~,~,el_right]=xlsread(expath,"RIGHT");

for i = 1:length(missing_el_inter)
    match=length(find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_inter(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_left(:,1),missing_el_inter(i))))))
        el_left(item,:)=[]
    end
    
    match=length(find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_inter(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_inter(i))))))
        el_right(item,:)=[]
    end
end

el_left=table(el_left,'VariableNames',{'Left'})
el_right=table(el_right,'VariableNames',{'Right'})
filename =folder+'\' +part+'_electrodes_sedoff.xlsx';
writetable(el_left,filename)
writetable(el_right,filename,'sheet',2)


%%%% Recovery
[~,~,el_left]=xlsread(expath,"LEFT");
[~,~,el_right]=xlsread(expath,"RIGHT");

for i = 1:length(missing_el_post)
    match=length(find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_post(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_post(i))))))
        el_left(item,:)=[]
    end
    
    match=length(find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_post(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_post(i))))))
        el_right(item,:)=[]
    end
end

el_left=table(el_left,'VariableNames',{'Left'})
el_right=table(el_right,'VariableNames',{'Right'})
filename =folder+'\' +part+'_electrodes_sedon2.xlsx';
writetable(el_left,filename)
writetable(el_right,filename,'sheet',2)

