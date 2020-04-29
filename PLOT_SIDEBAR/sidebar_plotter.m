
% Example, select 'EXAMLPE_images to modify'
waitfor(msgbox('Select the image data folder.'));
imagefolder=uigetdir(path)


waitfor(msgbox('Select the modified excel file.'));
exceldir = {uigetfile('*.xlsx')};

%%%%%%%%%%%%%%%%%%%%%%   START         %%%%%%%%%%%%%%%%%%%%
cd(imagefolder)

[~,~,el_left]=xlsread(string(exceldir),"LEFT");
[~,~,el_right]=xlsread(string(exceldir),"RIGHT");
E = dir;
E = E(ismember({E.name}, {'LEFT', 'RIGHT'}));
    
for i = 1:numel(E)
    if E(i).name=="LEFT"
        electrodes=el_left;
    end

    if E(i).name=="RIGHT"
        electrodes=el_right;
    end

    cd(imagefolder +"\"+ E(i).name)
    F = dir;
    F = F(ismember({F.name}, {'Directed Phase Lag Index', 'Phase Lag Index'}));
    

    for j = 1:numel(F)
        cd(imagefolder +"\"+ E(i).name+"\"+F(j).name)
        subfolder=imagefolder +"\"+ E(i).name+"\"+F(j).name;
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
   

