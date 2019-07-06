function [type,epoch,frequency] = get_content(file_name)
        content  = strsplit(file_name,'_');
        type = content{1};
        epoch = content{2};
        
        content = strsplit(content{3},'.');
        frequency = content{1};
end