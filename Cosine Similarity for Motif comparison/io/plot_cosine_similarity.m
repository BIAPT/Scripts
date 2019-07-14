function plot_cosine_similarity(similarities,title_string,x_labels,y_labels)
%   PLOT_COSINE_SIMILARITY shows a colormap of the cosine similarities
%   similarities: is a L*M matrix obtained using matrix_cosine_similarity
%   title_string: is the title of the plot
%   x_labels: is a cell array of labels of size L
%   y_labels: is a cell array of labels of size M

    %% Error Checking
    if(length(x_labels) ~= size(similarities,1) || length(y_labels) ~= size(similarities,2))
       error("Mismatch between labels and matrix size.");
    end
    
    %% Plotting
    figure;
    colormap('jet')
    imagesc(similarities);
    title(title_string);
    xlabel('Epoch') 
    ylabel('Epoch')
    xticks(1:length(x_labels))
    yticks(1:length(y_labels))
    xticklabels(x_labels)
    yticklabels(y_labels)
    caxis([0,1])
    colorbar;
end
