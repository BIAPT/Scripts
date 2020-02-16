function [x1,x2,y1,y2,Fig_circle]= default_circle(visibility)
    
    f_orange=[1 128/255 0/255];
    c_blue=[0 122/255 204/255];
    p_yellow=[1 201/255 51/255];
    o_green=[76/255 153/255 0];
    t_grey=[160/255 160/255 160/255];

    M = 10000 ;
    N = M/100 ;   % 10 regions (5 @each side)
    R1 = 0.9 ;% Radius of circle1
    R2 = 1.0 ; % Radius of circle 2
    th = linspace(0,2*pi,M) ;  % Angle 0 to 360
    % polar coordinates
    x1 = R1*cos(th) ; y1 = R1*sin(th) ;  
    x2 = R2*cos(th) ; y2 = R2*sin(th) ;

    %%Arrnage cooridnates into 10 colored parts
    el_frontal=13;
    el_central=13;
    el_parietal=9;
    el_occipital=5;
    el_temporal=8; 

    X1 = reshape(x1,[N,100])' ;
    Y1 = reshape(y1,[N,100])' ;
    X2 = fliplr(reshape(x2,[N,100])') ;
    Y2 = fliplr(reshape(y2,[N,100])' );

    % fill color
    Fig_circle = figure('visible',visibility);
    hold on
    
    for i =1+1:el_frontal+1
        patch([X1(i,:) X2(i,:)],[Y1(i,:) Y2(i,:)],f_orange,'edgecolor','none')    
        patch([X1(101-i,:) X2(101-i,:)],[Y1(101-i,:) Y2(101-i,:)],f_orange,'edgecolor','none')
    end

    for i =el_frontal+1+1:el_frontal+el_central+1
        patch([X1(i,:) X2(i,:)],[Y1(i,:) Y2(i,:)],c_blue,'edgecolor','none')    
        patch([X1(101-i,:) X2(101-i,:)],[Y1(101-i,:) Y2(101-i,:)],c_blue,'edgecolor','none')    
    end

    for i =el_frontal+el_central+1+1:el_parietal+el_frontal+el_central+1
        patch([X1(i,:) X2(i,:)],[Y1(i,:) Y2(i,:)],p_yellow,'edgecolor','none')    
        patch([X1(101-i,:) X2(101-i,:)],[Y1(101-i,:) Y2(101-i,:)],p_yellow,'edgecolor','none')    
    end

    for i =el_frontal+el_central+el_parietal+1+1:el_frontal+el_central+el_parietal+el_occipital+1
        patch([X1(i,:) X2(i,:)],[Y1(i,:) Y2(i,:)],o_green,'edgecolor','none')    
        patch([X1(101-i,:) X2(101-i,:)],[Y1(101-i,:) Y2(101-i,:)],o_green,'edgecolor','none')    
    end

    for i =el_frontal+el_central+el_parietal+el_occipital+1+1:el_frontal+el_central+el_parietal+el_occipital+el_temporal+1
        patch([X1(i,:) X2(i,:)],[Y1(i,:) Y2(i,:)],t_grey,'edgecolor','none')
        patch([X1(101-i,:) X2(101-i,:)],[Y1(101-i,:) Y2(101-i,:)],t_grey,'edgecolor','none')    
    end
    
    hold on
    plot(-1:1,0*ones(3),'white','linewidth',2)
    axis equal
    xlim([-1.2 1.2])
    ylim([-1.2 1.2])
    hold on 
    
end