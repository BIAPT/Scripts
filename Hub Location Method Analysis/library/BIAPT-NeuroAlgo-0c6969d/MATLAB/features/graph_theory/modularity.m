function [community] = modularity(matrix)
%MODULARITY Summary of this function goes here
%   matrix: a N*N binary or weighted square matrix
%   null_networks: 3d matrix containing pre-made null_networks
%   gamma: if large will detect smaller module, if small will detect larger
%   module

    [~,community] = community_louvain(matrix,1,[],'negative_sym'); % community, modularity
end


