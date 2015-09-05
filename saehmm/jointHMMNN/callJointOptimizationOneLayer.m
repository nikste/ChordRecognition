function [ output_args ] = callJointOptimizationOneLayer(  ffts_train_songs,gts_full, inits,transitions,mus,sigmas,mixmats,nn,ffts_test_no_preprocessing,gt_test,gt_test_lab)
%CALLJOINTOPTIMIZATIONONELAYER Summary of this function goes here
%   Detailed explanation goes here
error_total = [];
f = figure;
train_error_avrg = [];
for i=1:20

    
%     figure
%     visualize(nn.W{1});
%     drawnow;
    
    tic
    [ mus,sigmas,mixmats,nn ,train_error] = jointOptimizationOneLayer( ffts_train_songs,gts_full, inits,transitions,mus,sigmas,mixmats,nn );
    %mixmats
    toc
    train_error_avrg = [train_error_avrg mean(train_error)];
    tic
    [ error ] = testJoint( nn,inits,transitions, mus,sigmas,mixmats,ffts_test_no_preprocessing,gt_test,gt_test_lab,0 );
    toc
    disp(strcat('iteration :',num2str(i)));    
    error_total = [error_total error];
    figure;
    plot(error_total)
    ylim([0 1])
    xlim([0 21])
    drawnow;
    figure;
    plot(train_error_avrg);
    xlim([0 21]);
    drawnow;

end

output_args = error_total;

end