classdef oneEuro < handle
    properties
        firstTime;
        mincutoff;
        beta;
        dcutoff;
        xfilt;
        dxfilt;
    end
    
    methods
        function obj = oneEuro()
            obj.firstTime = true;
            obj.mincutoff = 1.0;
            obj.beta = 0.0;
            obj.dcutoff = 1.0;
            obj.xfilt = lpFilter;
            obj.dxfilt = lpFilter;
        end
        
        function y = filter(obj,x,rate)
            if (obj.firstTime)
                obj.firstTime = false;
                dx = 0;
            else
                dx = (x-obj.xfilt.last)*rate;
            end
            
            edx = obj.dxfilt.filter(dx, obj.alphafcn(rate, obj.dcutoff));
            cutoff = obj.mincutoff+obj.beta*abs(edx);
            y = obj.xfilt.filter(x, obj.alphafcn(rate, cutoff));
        end
        
        function y = alphafcn(~, rate, cutoff)
            tau = 1.0/(2*pi*cutoff);
            te = 1.0/rate;
            y = 1.0/(1.0+tau/te);
        end
    end
end