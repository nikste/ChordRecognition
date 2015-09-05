function [ error ] = testJoint( nn,inits,transitions, mus,sigmas,mixmats,ffts_test_no_preprocessing,gt_test,gt_test_lab,multi_res )
%TESTJOINT tests a joint model of neural network and gmm hmm.
% input: neural network, gmm hmm parameters, ffts for training, ground
% truth frame wise, ground truth continuous, multi resolution.

%% do whitening stuff like we did before
disp('transforming input')
pcps_test = {};
tic
for song = 1:length(gt_test)

    % pca: BAD, your results are bad and you should feel bad!
%     ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song}));
%     ffts_test_no_preprocessing{song} = bsxfun(@minus,ffts_test_no_preprocessing{song},m);%mean(ffts_test_no_preprocessing{song}));
%     ffts_test_no_preprocessing{song} = normr(ffts_test_no_preprocessing{song}*COEFF);
%     f = ffts_test_no_preprocessing{song};
%     ffts_test_no_preprocessing{song} = f(:,1:1200);

    %multires
    if (multi_res == 1)
        ffts_test_no_preprocessing{song} = createMultiResolutionFFT(ffts_test_no_preprocessing{song});
    else
        ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song}));
    end
    pcps_test{song} = runNN(nn,ffts_test_no_preprocessing{song});
    %till now best performing:
    %ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song})); 
   
    % also not so good:
    %ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song}));

% its okay   
%     ffts_test_no_preprocessing{song} = sqrt(ffts_test_no_preprocessing{song});
%     W =bsxfun(@minus,ffts_test_no_preprocessing{song},M) * P;%ffts_test_no_preprocessing{song}; 
%     ffts_test_no_preprocessing{song} = normr(W);

%performs bad
%     ffts_test_no_preprocessing{song} = preprocessData(ffts_test_no_preprocessing{song})
end
toc

%% recompute features with neural network

%% apply GMM HMM
[error] = testGMMHMM(inits,transitions, mus,sigmas,mixmats,pcps_test,gt_test,gt_test_lab)

end


%% runNN
% runs the neural network with input
% computes the output in chord label and as 1 of K representation
function [res] = runNN(nn,x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    res = nn.a{end};
    %[~, i] = max(nn.a{end},[],2);
    %label = i - 1;%because chords.
end