
function [] = modify_images(imagepath,lower, upper,electrodes,nr)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
fig1=openfig(imagepath);
caxis([lower(1) upper(1)]); %from 1.5 to 4
newname=erase(imagepath,".fig");
saveas(fig1,newname + nr + ".png")
disp("figure saved")


F= length(find(not(cellfun('isempty',(strfind(electrodes(:,2), 'F'))))));
C= length(find(not(cellfun('isempty',(strfind(electrodes(:,2), 'C'))))));
P= length(find(not(cellfun('isempty',(strfind(electrodes(:,2), 'P'))))));
O= length(find(not(cellfun('isempty',(strfind(electrodes(:,2), 'O'))))));
T= length(find(not(cellfun('isempty',(strfind(electrodes(:,2), 'T'))))));


i1=imread(newname+ nr + ".png");
image(i1)
hold on

Pixelsize=632/(F+C+P+O+T)

position =  [110+(Pixelsize*F/2) 740;...
    110+Pixelsize*F+(Pixelsize*C/2) 740;...
    110+Pixelsize*F+Pixelsize*C+((Pixelsize*P)/2) 740;...
    110+Pixelsize*F+Pixelsize*C+Pixelsize*P+((Pixelsize*O)/2) 740;...
    110+Pixelsize*F+Pixelsize*C+Pixelsize*P+Pixelsize*O+((Pixelsize*T)/2) 740];

RGB = insertText(i1,position(1,:),'F','FontSize',30,...
    'TextColor',[255 128 0],'BoxOpacity',0,'Font','Arial Bold');
RGB = insertText(RGB,position(2,:),'C','FontSize',30,...
    'TextColor',[0 122 204],'BoxOpacity',0,'Font','Arial Bold');
RGB = insertText(RGB,position(3,:),'P','FontSize',30,...
    'TextColor',[255 201 51],'BoxOpacity',0,'Font','Arial Bold');
RGB = insertText(RGB,position(4,:),'O','FontSize',30,...
    'TextColor',[76 153 0],'BoxOpacity',0,'Font','Arial Bold');
RGB = insertText(RGB,position(5,:),'T','FontSize',30,...
    'TextColor',[160 160 160],'BoxOpacity',0,'Font','Arial Bold');
hold on 

imshow(RGB)

%Vertical
rectangle('Position',[70,60,40,Pixelsize*F],...
         'LineWidth',1,'FaceColor',[1 128/255 0/255],'EdgeColor',[1 128/255 0/255])
rectangle('Position',[70,60+Pixelsize*F,40,Pixelsize*C],...
         'LineWidth',1,'FaceColor',[0 122/255 204/255],'EdgeColor',[0 122/255 204/255])
rectangle('Position',[70,60+Pixelsize*F+Pixelsize*C,40,Pixelsize*P],...
         'LineWidth',1,'FaceColor',[1 201/255 51/255],'EdgeColor',[1 201/255 51/255])
rectangle('Position',[70,60+Pixelsize*F+Pixelsize*C+Pixelsize*P,40,Pixelsize*O],...
         'LineWidth',1,'FaceColor',[76/255 153/255 0],'EdgeColor',[76/255 153/255 0])
rectangle('Position',[70,60+Pixelsize*O+Pixelsize*F+Pixelsize*C+Pixelsize*P,40,Pixelsize*T],...
         'LineWidth',1,'FaceColor',[160/255 160/255 160/255],'EdgeColor',[160/255 160/255 160/255])
hold on


Pixelsize=655/(F+C+P+O+T)

     % HORIZONTAL
rectangle('Position',[110,696,Pixelsize*F,40],...
         'LineWidth',1,'FaceColor',[1 128/255 0/255],'EdgeColor',[1 128/255 0/255])
rectangle('Position',[110+Pixelsize*F,696,Pixelsize*C,40],...
         'LineWidth',1,'FaceColor',[0 122/255 204/255],'EdgeColor',[0 122/255 204/255])
rectangle('Position',[110+Pixelsize*F+Pixelsize*C,696,Pixelsize*P,40],...
         'LineWidth',1,'FaceColor',[1 201/255 51/255],'EdgeColor',[1 201/255 51/255])
rectangle('Position',[110+Pixelsize*F+Pixelsize*C+Pixelsize*P,696,Pixelsize*O,40],...
         'LineWidth',1,'FaceColor',[76/255 153/255 0],'EdgeColor',[76/255 153/255 0])
rectangle('Position',[110+Pixelsize*O+Pixelsize*F+Pixelsize*C+Pixelsize*P,696,Pixelsize*T,40],...
         'LineWidth',1,'FaceColor',[160/255 160/255 160/255],'EdgeColor',[160/255 160/255 160/255])

saveas(fig1,newname + nr + ".png")
close(fig1)
end



