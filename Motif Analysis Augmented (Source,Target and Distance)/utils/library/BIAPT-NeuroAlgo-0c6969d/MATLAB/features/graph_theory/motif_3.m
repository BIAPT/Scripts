function [frequency, source, target, distance] = motif_3(network,channels_location, number_rand_network, bin_swaps, weight_frequency)
%MOTIF_3    
    
    %% 1) Calculate the motif for our network of interest.
    [intensity,coherence,frequency,source,target,distance] = motif3funct_wei_augmented(network, channels_location);
    
    % Create the matrices
    rand_intensity = zeros(number_rand_network,13,length(network));
    rand_coherence = zeros(number_rand_network,13,length(network));
    rand_frequency = zeros(number_rand_network,13,length(network)); 
    
    % Create X random network using our network of interest 
    parfor i = 1:number_rand_network
        [rand_network,~] = null_model_dir_sign(network,bin_swaps,weight_frequency);
        % Calculate the motif for the X random network. (BOTTLE NECK)
        [rand_intensity(i,:,:),rand_coherence(i,:,:),rand_frequency(i,:,:)] = motif3funct_wei_augmented(rand_network,channels_location);
    end
    
    %% 4) Calculate the Z score for each motifs
    
    % Might want to refactor this
    cat_rand_intensity = rand_intensity(1,:,:);
    cat_rand_coherence = rand_coherence(1,:,:);
    cat_rand_frequency = rand_frequency(1,:,:);
    for i = 2:number_rand_network
        cat_rand_intensity = cat(3,cat_rand_intensity,rand_intensity(i,:,:));
        cat_rand_coherence = cat(3,cat_rand_coherence,rand_coherence(i,:,:));
        cat_rand_frequency = cat(3,cat_rand_frequency,rand_frequency(i,:,:));
    end
    cat_rand_intensity = squeeze(cat_rand_intensity)';    
    cat_rand_coherence = squeeze(cat_rand_coherence)';    
    cat_rand_frequency = squeeze(cat_rand_frequency)';

    z_intensity = (mean(intensity') - mean(cat_rand_intensity)) ./ std(cat_rand_intensity);    
    z_coherence = (mean(coherence') - mean(cat_rand_coherence)) ./ std(cat_rand_coherence);
    z_frequency = (mean(frequency') - mean(cat_rand_frequency)) ./ std(cat_rand_frequency);
    
    
    % Here we set to 0 whatever is not significant at p < 0.05 (this should
    % be a parameter though)
    for i=1:13
        % Need to set the count for these motifs to 0 
        if(z_frequency(i) < 1.96) 
           for j=1:length(frequency)
              frequency(i,j) = 0;
           end
        end
    end
end