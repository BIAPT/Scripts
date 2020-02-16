function [pe, normalized_pe] = permutation_entropy(eeg_data, embedding_dimension, time_lag)
% permutation entropy for multivariate signals (just a repetition of calculation for multi-channel signals)
% 2018.1.19. Heonsoo Lee
% 2018.12.31 Yacine Mahdid (Refactored for readability)
%   Input:
%       eeg_data: data to calculate the measures on
%       embedding_dimension & time_delay: params for the pe code call (TODO
%       find what these parameters are doing)
%   Output:
%       the normalized permutation entropy
    
    %% Initialization of Variables
    eeg_data = eeg_data';
    numpat=factorial(embedding_dimension); % number of patterns
    denom=log2(numpat); % normalization factor
    Ddata=delay_reconstruction(eeg_data, time_lag, embedding_dimension); % delay reconstruction
    [~,number_channels]=size(eeg_data); % get number of channels
   
    %% Pre-allocation of Output
    pe = zeros(1,number_channels);
    normalized_pe = zeros(1,number_channels);
    
    for c=1:number_channels
        %% real value -> symoblic pattern
        udata=zeros(size(Ddata(:,:,c),1), 1);
        [~, Ddata_sort]=sort(Ddata(:,:,c),2);
        for i=1:embedding_dimension
            udata=udata+Ddata_sort(:,i)*10^(round(embedding_dimension/2)-i)'; % *** double precision: 10^-16
        end
        %% probability
        [u_patterns, ~]=unique(udata); % find unique patterns
        u_len=length(u_patterns); % # of unique patterns
        p=zeros(1, numpat);
        for u=1:u_len
            ind=find(udata==u_patterns(u));
            p(u)=length(ind)/size(udata,1);
        end
        pe(1,c)=-sum(p.*log2(p+eps));
        normalized_pe(1,c)=pe(c)./denom;
        clear p;
    end
  
end

function output=delay_reconstruction(data, delay_time, embedding_dimension)
% INPUT
% data: 2-d matrix (time by channel)
% delay_time: delay time
% embedding_dimension: embedding dimension
%
% OUTPUT: 3-d matrix (time by EMB vector by channel)
% output(1,:,1) = [data(1) data(1+L) data(1+2L) ... data(1+(m-1)L)]
%
% Heonsoo Lee 2011.11.17
% Yacine Mahdid 2018.12.31 (Refactored for readability)

    %% Initialize variables
    [length,number_channels]=size(data);
    output=zeros(length-delay_time*(embedding_dimension-1),embedding_dimension,number_channels);

    for current_channel=1:number_channels
        for current_dimension=1:embedding_dimension
            % Get the start and end times
            start_time = 1+(current_dimension-1)*delay_time;
            end_time = length-(embedding_dimension-current_dimension)*delay_time;
            % slice the data
            current_data = data(start_time:end_time, current_channel);
            % store it at the right place
            output(:, current_dimension, current_channel)= current_data;
        end
    end
end