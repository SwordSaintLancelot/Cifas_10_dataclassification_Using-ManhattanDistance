function [ model ] = cifar_10_evaluate( pred,gt )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
model = fitcknn(pred,gt,'NumNeighbors',1);
end

