function [] = callPerformTestBillboard7Inv(multi_res,pcp_pretrain,zeroMaskFrac,noisetype,vari,sparsity,sparsityPenalty,L2weightPenalty,dropout,id,shrinkfactor,layer1,layer2,layer3,layer4,layer5,layer6)

   
    
    %matlabpool('open',8);
    %display command line arguments:

    disp(strcat('layer1:',layer1));
    disp(strcat('layer2:',layer2));
    disp(strcat('layer3:',layer3));
    disp(strcat('layer4:',layer4));
    disp(strcat('layer5:',layer5));
    disp(strcat('layer6:',layer6));
    layer1 = str2num(layer1);
    layer2 = str2num(layer2);
    layer3 = str2num(layer3);
    layer4 = str2num(layer4);
    layer5 = str2num(layer5);
    layer6 = str2num(layer6);
    
    disp(strcat('multi resolution SDAE:',multi_res));
	disp(strcat('pcp intermediate targe:',pcp_pretrain));
    disp(strcat('zeroMaskFrac:',zeroMaskFrac));
    disp(strcat('noise type:',noisetype));
    disp(strcat('variance:',vari));
    disp(strcat('sparsity:',sparsity));
    disp(strcat('sparsityPenalty:',sparsityPenalty));
    disp(strcat('weightPenaltyL2:',L2weightPenalty));
    disp(strcat('dropout:',dropout));
    disp(strcat('id:',id));
    disp(strcat('shrinkfactor:',shrinkfactor));
	%convert command line arguments:

    shrinkfactor = str2num(shrinkfactor);
	multi_res = str2num(multi_res);
	pcp_pretrain = str2num(pcp_pretrain);

    zeroMaskFrac = str2num(zeroMaskFrac);
    vari = str2num(vari);
    sparsity = str2num(sparsity);
    dropout = str2num(dropout);
    sparsityPenalty = str2num(sparsityPenalty);
    L2weightPenalty = str2num(L2weightPenalty);
	%first_layer = 200;
	%second_layer = 0;
	%multi_res = 0;
	%pcp_pretrain = 0;
	%[ output_args ] = performTest(first_layer,second_layer, multi_res, pcp_pretrain,id)
    
    [ffts_train,gts_full,gts_train] =loadTrainSAEBillboard7Inv(shrinkfactor,multi_res,id); %loadTrainSAEBillboard(shrinkfactor,multi_res,id);%%loadTrainSAEBillboard(shrinkfactor,0,'a');
    

%if (nargin ==11)
%zeroMaskFrac = varargin{1};
%sparsity = varargin{2};
%dropout = varargin{3};
%sparsityPenalty = varargin{4};
    if(strcmp(noisetype,'isoGauss'))
        [nn,inits,transitions]= trainSAEHMMBillboardIsoGauss7Inv(ffts_train,gts_train,gts_full,layer1,layer2,layer3,layer4,layer5,layer6, multi_res, pcp_pretrain,vari,zeroMaskFrac,sparsity,dropout,sparsityPenalty,L2weightPenalty);
    else
        [nn,inits,transitions]= trainSAEHMMBillboard7Inv(ffts_train,gts_train,gts_full,layer1,layer2,layer3,layer4,layer5,layer6, multi_res, pcp_pretrain,zeroMaskFrac,sparsity,dropout,sparsityPenalty,L2weightPenalty);
    end  
    ffts_train = [];
    gts_full = [];
    gts_train = [];
    
     [gt_test,ffts_test_no_preprocessing] = loadTestSAEHMMBillboard7Inv(id);%loadTestSAEHMM(id);
     [ gt_test_lab ] = loadLabTestBillboard7Inv( id);%loadLabTest(id);
     
     [error] = testSAEHMM(nn,inits,transitions,ffts_test_no_preprocessing,gt_test,gt_test_lab,multi_res);
     output_args = error;
	disp(mat2str(output_args));


end