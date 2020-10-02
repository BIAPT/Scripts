%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Plot colored bars around pli and dPLI images                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%     Choose following variables: 
part= "NET-ICU-002-MG"

%%%%%%%%%%%%%%%%%%%%%%   START         %%%%%%%%%%%%%%%%%%%%
folder = 'C:\Users\BIAPT\Desktop\'+part
cd(folder)

D = dir;
D = D(~ismember({D.name}, {'Electrode number and region.xlsx','.', '..'}));
D(2)
k=1 
for k = 1:numel(D)
    cd(folder +"\"+ D(k).name)
    ex = dir(fullfile((folder +"\"+ D(k).name+"\"), 'WSAS*'));
    expath=folder +"\"+ D(k).name+"\"+ ex(1).name
    [~,~,el_left]=xlsread(expath,"Tabelle1");
    [~,~,el_right]=xlsread(expath,"Sheet2");
    E = dir;
    E = E(ismember({E.name}, {'LEFT', 'RIGHT'}));
    
    for i = 1:numel(E)
        if E(i).name=="LEFT"
            electrodes=el_left;
        end
        
        if E(i).name=="RIGHT"
            electrodes=el_right;
        end
        
        cd(folder +"\"+ D(k).name+"\"+ E(i).name)
        F = dir;
        F = F(ismember({F.name}, {'Directed Phase Lag Index', 'Phase Lag Index'}));
        
        for j = 1:numel(F)
            cd(folder +"\"+ D(k).name+"\"+ E(i).name+"\"+F(j).name)
            subfolder=folder +"\"+ D(k).name+"\"+ E(i).name+"\"+F(j).name;
            imgfiles = dir(fullfile(subfolder, '*.fig'));
            
            if F(j).name == "Directed Phase Lag Index"
                lower=[0.3 0.35 0.4 0.45];
                upper=[0.7 0.65 0.6 0.55];
                for l=1:length(lower)
                    for im=1:length(imgfiles)
                        mod_im=imgfiles(im).folder+"\"+(imgfiles(im).name)
                        modify_images(mod_im,lower(l),upper(l),electrodes,l)
                    end
                end
                
            end
                
            if F(j).name == "Phase Lag Index"
                lower=[0 0 0 0];
                upper=[0.1 0.15 0.2 0.25];
                for l=1:length(lower)
                    for im=1:length(imgfiles)
                        mod_im=imgfiles(im).folder+"\"+(imgfiles(im).name)
                        modify_images(mod_im,lower(l),upper(l),electrodes,l)
                    end
                end
            end
        end
        
    end
   
end



