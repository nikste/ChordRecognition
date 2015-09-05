function [ ffts_train,gts_full,gts_train,ffts_train_songs,nn,inits,transitions, mus,sigmas,mixmats, gt_test,ffts_test_no_preprocessing,gt_test_lab] = callJointOptimizationOneLayer_init(  )
%CALLJOINTOPTIMIZATIONONELAYER Summary of this function goes here
%   Detailed explanation goes here
[ffts_train,gts_full,gts_train,ffts_train_songs] = loadTrainSAEJOINT(32,0,'a');

[ chords_sorted ] = sortChords( ffts_train, gts_train );

[nn,inits,transitions, mus,sigmas,mixmats] = trainJointPre(chords_sorted,gts_full,200,0, 0, 0);

[gt_test,ffts_test_no_preprocessing] = loadTestSAEHMM('a');
[ gt_test_lab ] = loadLabTest('a');
end