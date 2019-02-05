function [ downfilename ] = downfilename(url, filename)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
downfilename = websave(filename,url);

end
