function []= plot_points_labels(mode,name,location,x_location,y_location,x_label,y_label)
    hold on
    plot(x_location, y_location, 'black.', 'LineWidth', 2, 'MarkerSize', 16);
    
    %Calculate Angle for label
    P0 = [0, 0];
    P1 = [1,0];
    P2 = [x_label, y_label];
    n1 = (P2 - P0) / norm(P2 - P0);  % Normalized vectors
    n2 = (P1 - P0) / norm(P1 - P0);
    ang = acos(dot(n1, n2));
    ang = ang*(180/pi);
    if y_label < 0
        ang=-ang;
    end
    
    if location == "mid" && mode == "Midline"
        h=text(x_label,y_label,name);
        set(h,'Rotation',0,'FontSize', 5);
    end

    if location ~= "mid"
        h=text(x_label,y_label,name);
        set(h,'Rotation',ang,'FontSize', 8);
    end
    
    
end

