function [ ] = closematlabpool( )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

restoredefaultpath
matlabpool('close');
addpath(genpath('/home/nikste/workspace-m/masterthesis_matlab'));
end
