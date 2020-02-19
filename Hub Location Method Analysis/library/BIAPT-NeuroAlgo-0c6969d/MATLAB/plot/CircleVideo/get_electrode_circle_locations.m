function [channel_data]= get_electrode_circle_locations(recording,x1,y1,x2,y2)
    
    ex = dir('Electrode number*');
    [~,~,el_left]=xlsread(ex.folder+"\"+ ex(1).name,"Left");
    [~,~,el_right]=xlsread(ex.folder+"\"+ ex(1).name,"Right");
    [~,~,el_midline]=xlsread(ex.folder+"\"+ ex(1).name,"MIDELINE");
    
    channel_data=[]

    
    for  i = 1:recording.number_channels
        name=recording.channels_location(i).labels;
        Index_l = find(endsWith(el_left(:,1),recording.channels_location(i).labels));
        Index_r = find(endsWith(el_right(:,1),recording.channels_location(i).labels));
        Index_m = find(endsWith(el_midline(:,1),recording.channels_location(i).labels));

        if length(Index_l)== 1 
            region="left";
            nr=round(cell2mat(el_left(Index_l,3)));
            X_location = x1(nr);
            y_location = y1(nr);
            X_label = x2(nr);
            y_label = y2(nr);
        end
        
        if length(Index_r)== 1
            region="right";
            nr=round(cell2mat(el_right(Index_r,3)));
            X_location = x1(nr);
            y_location = y1(nr);
            X_label = x2(nr);
            y_label = y2(nr);
        end

        if length(Index_m)== 1
            region= "mid";
            X_location = cell2mat(el_midline(Index_m,3));
            y_location = 0;
            X_label = X_location+0.02;
            y_label = 0;
        end
        
        table_i=table({name},{region},{X_location},{y_location},{X_label},{y_label});
        channel_data = [channel_data;table_i];
        
    end
    
    channel_data.Properties.VariableNames = {'name','region','X_location','y_location','X_label','y_label'};
end

    