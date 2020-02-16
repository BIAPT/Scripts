function []=plot_connectivity_line(matrix,el_data,maxvalue, threshold, mode)
    
    col_scale=jet(100);
    
    if mode == "Intra"
        for c =2:length(matrix) %column of matrix
            for r = 1:c-1       %row of matrix
                if matrix(c,r) > threshold && ...
                        ((string(el_data.region(c))== "left" && string(el_data.region(r))== "left")||...
                        (string(el_data.region(c))== "right" && string(el_data.region(r))== "right"))
                    % both electrodes come from the same hemisphere and are
                    % higher connected then the threshold
                    X_row=cell2mat(el_data.X_location(r));
                    X_col=cell2mat(el_data.X_location(c));
                    Y_row=cell2mat(el_data.y_location(r));
                    Y_col=cell2mat(el_data.y_location(c));
                    value = matrix(c,r)/maxvalue;
                    plot([X_row X_col],[Y_row Y_col],'Color',col_scale(round(value*100),:),'lineWidth',1)
                    hold on
                end
            end
        end
    end 

   
    if mode == "Inter"
        for c =2:length(matrix) %column of matrix
            for r = 1:c-1       %row of matrix
                if matrix(c,r) > threshold && ...
                        ((string(el_data.region(c))== "left" && string(el_data.region(r))== "right")||...
                        (string(el_data.region(c))== "right" && string(el_data.region(r))== "left"))
                    % both electrodes come from the same hemisphere and are
                    % higher connected then the threshold
                    X_row=cell2mat(el_data.X_location(r));
                    X_col=cell2mat(el_data.X_location(c));
                    Y_row=cell2mat(el_data.y_location(r));
                    Y_col=cell2mat(el_data.y_location(c));
                    value = matrix(c,r)/maxvalue;
                    plot([X_row X_col],[Y_row Y_col],'Color',col_scale(round(value*100),:),'lineWidth',1)
                    hold on
                end
            end
        end
    end
    
    if mode == "Midline"
        for c =2:length(matrix) %column of matrix
            for r = 1:c-1       %row of matrix
                if matrix(c,r) > threshold && ...
                        ((string(el_data.region(c))== "mid" || string(el_data.region(r))== "mid")&&...
                        (string(el_data.region(c))~= "mid" || string(el_data.region(r))~= "mid"))
                    % both electrodes come from the same hemisphere and are
                    % higher connected then the threshold
                    X_row=cell2mat(el_data.X_location(r));
                    X_col=cell2mat(el_data.X_location(c));
                    Y_row=cell2mat(el_data.y_location(r));
                    Y_col=cell2mat(el_data.y_location(c));
                    value = matrix(c,r)/maxvalue;
                    plot([X_row X_col],[Y_row Y_col],'Color',col_scale(round(value*100),:),'lineWidth',1)
                    hold on
                end
            end
        end
    end
    
end