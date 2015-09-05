function [ ] = openmatlabpool( )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

restoredefaultpath
matlabpool('open',8);
addpath(genpath('/home/nikste/workspace-m/masterthesis_matlab'));
end

