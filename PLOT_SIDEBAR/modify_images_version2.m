function [] = modify_images_version2(imagepath,lower, upper,electrodes,nr)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
fig1=openfig(imagepath);
caxis([lower(nr) upper(nr)]); %from 1.5 to 4
newname=erase(imagepath,".fig");
saveas(fig1,newname + nr + ".png")
disp("figure saved")

F= electrodes(1);
C= electrodes(2);
P= electrodes(3);
O= electrodes(4);
T= electrodes(5);

i1=imread(newname+ nr + ".png");
image(i1)
hold on

Pixelsize=600/(F+C+P+O+T)

position =  [90+(Pixelsize*F/2) 610;...
    90+Pixelsize*F+(Pixelsize*C/2) 610;...
    90+Pixelsize*F+Pixelsize*C+((Pixelsize*P)/2) 610;...
    90+Pixelsize*F+Pixelsize*C+Pixelsize*P+((Pixelsize*O)/2) 610;...
    90+Pixelsize*F+Pixelsize*C+Pixelsize*P+Pixelsize*O+((Pixelsize*T)/2) 610];

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

Pixelsize=535/(F+C+P+O+T)
%Vertical
rectangle('Position',[60,50,40,Pixelsize*F],...
         'LineWidth',1,'FaceColor',[1 128/255 0/255],'EdgeColor',[1 128/255 0/255])
rectangle('Position',[60,50+Pixelsize*F,40,Pixelsize*C],...
         'LineWidth',1,'FaceColor',[0 122/255 204/255],'EdgeColor',[0 122/255 204/255])
rectangle('Position',[60,50+Pixelsize*F+Pixelsize*C,40,Pixelsize*P],...
         'LineWidth',1,'FaceColor',[1 201/255 51/255],'EdgeColor',[1 201/255 51/255])
rectangle('Position',[60,50+Pixelsize*F+Pixelsize*C+Pixelsize*P,40,Pixelsize*O],...
         'LineWidth',1,'FaceColor',[76/255 153/255 0],'EdgeColor',[76/255 153/255 0])
rectangle('Position',[60,50+Pixelsize*O+Pixelsize*F+Pixelsize*C+Pixelsize*P,40,Pixelsize*T],...
         'LineWidth',1,'FaceColor',[160/255 160/255 160/255],'EdgeColor',[160/255 160/255 160/255])
hold on


Pixelsize=600/(F+C+P+O+T)

     % HORIZONTAL
rectangle('Position',[104,585,Pixelsize*F,35],...
         'LineWidth',1,'FaceColor',[1 128/255 0/255],'EdgeColor',[1 128/255 0/255])
rectangle('Position',[104+Pixelsize*F,585,Pixelsize*C,35],...
         'LineWidth',1,'FaceColor',[0 122/255 204/255],'EdgeColor',[0 122/255 204/255])
rectangle('Position',[104+Pixelsize*F+Pixelsize*C,585,Pixelsize*P,35],...
         'LineWidth',1,'FaceColor',[1 201/255 51/255],'EdgeColor',[1 201/255 51/255])
rectangle('Position',[104+Pixelsize*F+Pixelsize*C+Pixelsize*P,585,Pixelsize*O,35],...
         'LineWidth',1,'FaceColor',[76/255 153/255 0],'EdgeColor',[76/255 153/255 0])
rectangle('Position',[104+Pixelsize*O+Pixelsize*F+Pixelsize*C+Pixelsize*P,585,Pixelsize*T,35],...
         'LineWidth',1,'FaceColor',[160/255 160/255 160/255],'EdgeColor',[160/255 160/255 160/255])

saveas(fig1,newname + nr + ".png")
close(fig1)
end



