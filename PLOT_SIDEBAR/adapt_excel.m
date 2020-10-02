%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                  Adapt excel files                           %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%     Choose following variables: 
%part="WSAS_19"
part="WSAS_22"


%20 missing_el_baseline=["E013","E006","E112","E105","E106","E007","E111","E080",...
%    "E118","E055","E061","E062","E079"]

%20 missing_el_anesthesia=["E013","E006","E112","E105","E106","E007","E005","E111","E118",...
%    "E012","E020","E079","E080","E087"]

%20 missing_el_recovery=["E013","E006","E112","E105","E106","E007","E111","E118",...
%    "E054"]
missing_el_baseline=[]
missing_el_anesthesia=[]
missing_el_recovery=[]



missing_el_baseline=["E010","E003","E004","E005","E118","E124","E123","E117",...
    "E094","E088","E127","E128","E043","E048","E049","E055","E035","E044","E029"]

missing_el_anesthesia=["E010","E003","E004","E005","E118","E124","E123","E117",...
    "E094","E088","E127","E128","E048","E049","E044","E043"]

missing_el_recovery=["E010","E003","E004","E005","E118","E124","E123","E117",...
    "E094","E088","E127","E128","E043","E048","E049"]

%%%%%%%%%%%%%%%%%%%%%%   START         %%%%%%%%%%%%%%%%%%%%
folder = 'C:\Users\User\Documents\1_MASTER\LAB\DOC\Data\'+part
cd(folder)

expath="C:\Users\User\Documents\1_MASTER\LAB\DOC\Material\Electrode number and region.xlsx"
[~,~,el_left]=xlsread(expath,"Left");
[~,~,el_right]=xlsread(expath,"Right");

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
filename =folder+'\' +part+'_electrodes_baseline.xlsx';
writetable(el_left,filename);
writetable(el_right,filename,'sheet',2)

%%%% Anesthesia
[~,~,el_left]=xlsread(expath,"Left");
[~,~,el_right]=xlsread(expath,"Right");

for i = 1:length(missing_el_anesthesia)
    match=length(find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_anesthesia(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_anesthesia(i))))))
        el_left(item,:)=[]
    end
    
    match=length(find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_anesthesia(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_anesthesia(i))))))
        el_right(item,:)=[]
    end
end

el_left=table(el_left,'VariableNames',{'Left'})
el_right=table(el_right,'VariableNames',{'Right'})
filename =folder+'\' +part+'_electrodes_anesthesia.xlsx';
writetable(el_left,filename)
writetable(el_right,filename,'sheet',2)


%%%% Recovery
[~,~,el_left]=xlsread(expath,"Left");
[~,~,el_right]=xlsread(expath,"Right");

for i = 1:length(missing_el_recovery)
    match=length(find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_recovery(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_left(:,1), missing_el_recovery(i))))))
        el_left(item,:)=[]
    end
    
    match=length(find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_recovery(i)))))))
    if match==1
        item=find(not(cellfun('isempty',(strfind(el_right(:,1), missing_el_recovery(i))))))
        el_right(item,:)=[]
    end
end

el_left=table(el_left,'VariableNames',{'Left'})
el_right=table(el_right,'VariableNames',{'Right'})
filename =folder+'\' +part+'_electrodes_recovery.xlsx';
writetable(el_left,filename)
writetable(el_right,filename,'sheet',2)

