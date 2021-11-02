classdef lpFilter < handle
    properties
        firstTime;
        hatxprev;
    end
    
    methods
        function obj = lpFilter()
            obj.firstTime = true;
        end
        
        function y = last(obj)
            y = obj.hatxprev;
        end
        
        function y = filter(obj, x, alphaval)
            if(obj.firstTime)
                obj.firstTime = false;
                hatx = x;
            else
                hatx = alphaval*x+(1-alphaval)*obj.hatxprev;
            end
            
            obj.hatxprev = hatx;
            
            y = hatx;
        end
    end
end