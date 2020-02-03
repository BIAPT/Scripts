classdef Result
    %RESULT Hold the output data along with meta data of the
    %recording
    
    properties
        % result information
        type; % type of result data structure we have (e.g. wpli)
        data; % where the result are stored
        metadata; % extra information to manipulate the data
        parameters; % parameters used to generate these result
    end
    
    methods
        function obj = Result(type, recording)
            %RESULT Construct an instance of this class
            %   accept a type and a recording which will be used to create
            %   the metada (copied over there)
            obj.type = type;
            obj.metadata = struct();
            obj.data = struct();
            obj.parameters = struct();
            
            obj.metadata.sampling_rate = recording.sampling_rate;
            obj.metadata.length_recording = recording.length_recording;
            obj.metadata.number_channels = recording.number_channels;
            obj.metadata.channels_location = recording.channels_location;
            obj.metadata.recording_creation_date = recording.creation_date;
        end
        
        function save(obj, filename, pathname)
            % SAVE: will create a struct version of the object which will
            % be saved on disk
            
            % Variables initialization  
            full_path = strcat(pathname,filesep,filename,".mat");
            
            % Create the struct and populating it
            result = struct();
            result.type = obj.type;
            result.metadata = obj.metadata;
            result.data = obj.data;
            result.parameters = obj.parameters;
            
            disp(strcat("Saving the Result under name ", filename, " at ", pathname));
            save(full_path,'result');
            
        end
        
        function make_video_topgraphic_map(obj, filename, pathname)
            %% This should only work for wPLI and dPLI
            if(strcmp(obj.type,"wpli") || strcmp(obj.type,"dpli"))
                full_path = strcat(pathname,filesep,filename);
                topographic_map = mean(obj.data.wpli,3);
                channels_location = obj.metadata.channels_location;
                step_size = obj.parameters.step_size; 
                make_video_topographic_map(full_path, topographic_map, channels_location, step_size)
            end
        end
        
        function make_video_functional_connectivity(obj, filename, pathname)
           % filename, functional_connectivity_matrices, step_size
        end
    end
end

