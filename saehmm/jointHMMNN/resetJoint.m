function [ nn,mixmats,mus,sigmas ] = resetJoint( nn,mixmats,mus,sigmas,nn_b,mus_b,mixmats_b,sigmas_b )
%RESETJOINT Summary of this function goes here
%   Detailed explanation goes here
disp('resetting model to pretrained parameters')
nn = nn_b;
mixmats = mixmats_b;
mus = mus_b;
sigmas = sigmas_b;
end

