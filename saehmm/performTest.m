function [ output_args ] = performTest(first_layer,second_layer, multi_res, pcp_pretrain,id)
%PERFORMTEST performs 10 fold cross validation on dataset
%   first_layer : number of hidden nodes in first layer (greater 0)
%   second_layer : number of hidden nodes in second layer (can be 0)
%   multi_res : constructs multiple resolution nn (4500 input nodes instead
%   of 1500
%   pcp_pretrain : adds additional intermediate layer with pcp pretraining with perfect pcp. 

output_arg = [];
for offset=6:10
    shrinkfactor = 10;
    disp(strcat('calculating',num2str(offset)));
    disp(strcat('shrinkfactor:',num2str(shrinkfactor)));
    [blacklist] = divideInTrainTest(offset,shrinkfactor,id);
    [ffts_train,gts_full,gts_train] = loadTrainSAE(2,multi_res,id);


    %[chords_present] =  testForAllChords(gts_train);
    %[gts_train,ffts_train] = decimateTrain(ffts_train,gts_train,chords_present,2);
    ffts_train = ffts_train(1:2:end,:);
    gts_train = gts_train(1:2:end,:);
    [chords_present] =  testForAllChords(gts_train)
    size(ffts_train)
    %ffts_train = normr(sqrt(ffts_train));
    [nn,inits,transitions]= trainSAEHMM(ffts_train,gts_train,gts_full,first_layer,second_layer, multi_res, pcp_pretrain);

    ffts_train = [];
    gts_full = [];
    gts_train = [];
    blacklist = [];

    [gt_test,ffts_test_no_preprocessing] = loadTestSAEHMM(id);

    [ gt_test_lab ] = loadLabTest(id );

    [error] = testSAEHMM(nn,inits,transitions,ffts_test_no_preprocessing,gt_test,gt_test_lab,multi_res)%,M,P,COEFF,m);

    disp(strcat('current error:',num2str(error)));

    gt_test = [];
    ffts_test_no_preprocessing = [];
    gt_test_lab = [];
    nn = [];
    inits = [];
    transitions = [];

    output_arg = [output_arg;error];
    output_arg
end


output_args = output_arg;

end

