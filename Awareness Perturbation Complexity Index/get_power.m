function [bandpass_power] = get_power(power, bandpass)
%GET POWER Average power at bandpass and return matrice for power at each
%channels

    % Initialize the empty topographic maps
    [number_maps, number_channels, ~] = size(power);
    bandpass_power = zeros(number_maps,number_channels);

    % Average the power at a specific bandpass
    for m = 1:number_maps
        for c = 1:number_channels

            % Average the power at this particular channel in our bandpass
            channel_power = power(m,c,bandpass(1):bandpass(2));
            bandpass_power(m,c) = mean(channel_power);

        end
    end
end

