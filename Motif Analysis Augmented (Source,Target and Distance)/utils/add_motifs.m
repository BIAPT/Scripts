function [avg_motifs, motifs_count] = add_motifs(avg_motifs, avg_channels_location,e_i, motifs, channels_location, motifs_count)

    for i=1:length(avg_channels_location)
        current_label = avg_channels_location(i).labels;
        is_found = 0;
        for j=1:length(channels_location)
           if(strcmp(channels_location(j).labels, current_label))
               is_found = j;
               break;
           end
        end
        
        if(is_found ~= 0)
            j = is_found;
            for m_i = 1:13
                % if the sum of the channels frequency is bigger than 0 it
                % means we have a significant motif
                if(sum(motifs.frequency(m_i,:)) > 0)
                    avg_motifs.frequency(e_i,m_i,i) = avg_motifs.frequency(e_i,m_i,i) + motifs.frequency(m_i,j);
                    avg_motifs.source(e_i,m_i,i) = avg_motifs.source(e_i,m_i,i) + motifs.source(m_i,j);
                    avg_motifs.distance(e_i,m_i,i) = avg_motifs.distance(e_i,m_i,i) + motifs.distance(m_i,j);
                    motifs_count(e_i,m_i,i) = motifs_count(e_i,m_i,i) + 1;
                end
            end
        end
    end
end