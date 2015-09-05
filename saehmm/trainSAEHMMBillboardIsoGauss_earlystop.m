function [nn,inits,transitions]= trainSAEHMMBillboardIsoGauss_earlystop(ffts_train,gts_train,gts_full,layer1,layer2,layer3,layer4,layer5,layer6, multi_res, pcp_pretrain,vari,varargin)
%[sae,nn,im1,im2,im3,err,inits,transitions] =
% additional parameters:
% zeroMaskedFrac
% sparsity
% dropout
% and or not and sparsityPenalty




zeroMaskFrac = 0.7;
sparsity = 0.05;
dropout = 0.5;
sparsityPenalty = 0;
L2weightPenalty = 0;
%% variable inputs has to be three ? (nonsparsityPenalty, weightPenaltyL2)
nargin
if (nargin > 0)
    
    if (nargin == 15)
        zeroMaskFrac = varargin{1};
        sparsity = varargin{2};
        dropout = varargin{3};
    end
    if (nargin ==16)
        zeroMaskFrac = varargin{1};
        sparsity = varargin{2};
        dropout = varargin{3};
        sparsityPenalty = varargin{4};
    end
    if (nargin == 17)
        zeroMaskFrac = varargin{1};
        sparsity = varargin{2};
        dropout = varargin{3};
        sparsityPenalty = varargin{4};       
        L2weightPenalty = varargin{5};
    end
end



%% initialize HMM
disp('initializing hmm')
[inits,transitions] = countHMMParams(gts_full);

disp(strcat('size of ground truths:',num2str(size(gts_train,1))));
disp(strcat('size of ffts:',num2str(size(ffts_train,1))));
%% train sae on shrunk dataset
% for safety reasons:
assert(size(gts_train,1) == size(ffts_train,1),'length of ffts and ground truth dont match!');


% does preprocessing internally
% preprocess data
%[ffts_train_,M,P] = preprocessData(ffts_train);
%ffts_train = sqrt(ffts_train);
%ffts_train = normr(ffts_train);
disp(strcat('layer1:',num2str(layer1)));
disp(strcat('layer2:',num2str(layer2)));
disp(strcat('layer3:',num2str(layer3)));
disp(strcat('layer4:',num2str(layer4)));
disp(strcat('layer5:',num2str(layer5)));
disp(strcat('layer6:',num2str(layer6)));
num_batch = 1000;
disp(strcat('num_batch:',num2str(num_batch)));
num_train_unsup = 30;%30%10;
disp(strcat('num_train_unsup:',num2str(num_train_unsup)));
num_train_pcp = 1000;%1000%150;
disp(strcat('num_train_pcp',num2str(num_train_pcp)));
num_train_chord = 1000;%1000%150;
disp(strcat('num_train_chord',num2str(num_train_chord)));

disp(strcat('sparsity_target:',num2str(sparsity)));
disp(strcat('nonSparsityPenalty:',num2str(sparsityPenalty)));
disp(strcat('weightPenaltyL2:',num2str(L2weightPenalty)));
disp(strcat('zeroMaskedFraction:',num2str(zeroMaskFrac)));
disp(strcat('using isotropic Gaussian noise variance:',num2str(vari)));
disp(strcat('dropout:',num2str(dropout)));
%randomize training samples
r = randperm(size(ffts_train,1));
ffts_train = ffts_train(r,:);
gts_train = gts_train(r,:);


if (multi_res == 1)
    if (pcp_pretrain == 1)
        nn = constructAEPCP(ffts_train,gts_train,layer1,layer2,layer3,layer4,layer5,layer6,num_batch,num_train_unsup,num_train_pcp,num_train_chord,4500,sparsity,dropout,zeroMaskFrac,sparsityPenalty,L2weightPenalty,vari);
    else
        nn = constructAE(ffts_train,gts_train,layer1,layer2,layer3,layer4,layer5,layer6,num_batch,num_train_unsup,num_train_pcp,num_train_chord,4500,sparsity,dropout,zeroMaskFrac,sparsityPenalty,L2weightPenalty,vari);%constructAE(ffts_train,gts_train,200,0); %constructAE(ffts_train,gts_train,200,0);
    end
else
    if (pcp_pretrain == 1)
        nn = constructAEPCP(ffts_train,gts_train,layer1,layer2,layer3,layer4,layer5,layer6,num_batch,num_train_unsup,num_train_pcp,num_train_chord,1500,sparsity,dropout,zeroMaskFrac,sparsityPenalty,L2weightPenalty,vari);
    else
        nn = constructAE(ffts_train,gts_train,layer1,layer2,layer3,layer4,layer5,layer6,num_batch,num_train_unsup,num_train_pcp,num_train_chord,1500,sparsity,dropout,zeroMaskFrac,sparsityPenalty,L2weightPenalty,vari);
    end
end
%% return trained machine.
disp('enjoy your result!')
%done


end


%% constructAE 
% constructs stacked autoencoder
% train_x : training data input (ffts)
% train_y : training data ground truth

function [nn] = constructAE(train_x,train_y,layer1,layer2,layer3,layer4,layer5,layer6,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs,sparsity,dropout,zeroMaskFrac,sparsityPenalty,L2weightPenalty,vari)
disp('constructing SAE')
% parameters

%truncate data
val_x = train_x(1:10:end,:);
val_y = train_y(1:10:end,:);


train_x(1:10:end,:) = [];
train_y(1:10:end,:) = [];
overflow = mod(size(train_x,1),num_batch);
val_x = [val_x; train_x((end-overflow):end,:)];
val_y = [val_y; train_y((end-overflow):end,:)];
train_x = train_x(1:(end - overflow),:);
train_y = train_y(1:(end - overflow),:);


%convert y in 1ofK
train_y = convertTo1K(train_y);
val_y = convertTo1K(val_y);
if(layer1 > 0 && layer2 == 0)
%train sae
    rand('seed',cputime)
    sae = saesetup([num_inputs layer1]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
        
    %options
    opts.numepochs = num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    %back prop
    nn = nnsetup([num_inputs layer1 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2                  = L2weightPenalty;
        
    % Train the NN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end
  
if(layer1 > 0 && layer2 > 0 && layer3 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    % back prop
    nn = nnsetup([num_inputs layer1 layer2 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2           = L2weightPenalty;
        
    % Train the NN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end

if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    % back prop
    nn = nnsetup([num_inputs layer1 layer2 layer3 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.W{3} = sae.ae{3}.W{1};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2           = L2weightPenalty;
    
    % Train the NN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end
if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 > 0 && layer5 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3 layer4]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{4}.activation_function       = 'sigm';%'sigm';
    sae.ae{4}.learningRate              = 1;
    sae.ae{4}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{4}.sparsityTarget            = sparsity;
    sae.ae{4}.dropoutFraction           = dropout;
    sae.ae{4}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{4}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    % back prop
    nn = nnsetup([num_inputs layer1 layer2 layer3 layer4 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.W{3} = sae.ae{3}.W{1};
    nn.W{4} = sae.ae{4}.W{1};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2           = L2weightPenalty;
    
    % Train the NN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end


if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 > 0 && layer5 > 0 && layer6 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3 layer4 layer5]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
        
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{4}.activation_function       = 'sigm';%'sigm';
    sae.ae{4}.learningRate              = 1;
    sae.ae{4}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{4}.sparsityTarget            = sparsity;
    sae.ae{4}.dropoutFraction           = dropout;
    sae.ae{4}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{4}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{5}.activation_function       = 'sigm';%'sigm';
    sae.ae{5}.learningRate              = 1;
    sae.ae{5}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{5}.sparsityTarget            = sparsity;
    sae.ae{5}.dropoutFraction           = dropout;
    sae.ae{5}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{5}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    % back prop
    nn = nnsetup([num_inputs layer1 layer2 layer3 layer4 layer5 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.W{3} = sae.ae{3}.W{1};
    nn.W{4} = sae.ae{4}.W{1};
    nn.W{5} = sae.ae{5}.W{1};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2                  = L2weightPenalty;    
    % Train the NN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end


if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 > 0 && layer5 > 0 && layer6 > 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3 layer4 layer5 layer6]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{4}.activation_function       = 'sigm';%'sigm';
    sae.ae{4}.learningRate              = 1;
    sae.ae{4}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{4}.sparsityTarget            = sparsity;
    sae.ae{4}.dropoutFraction           = dropout;
    sae.ae{4}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{4}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{5}.activation_function       = 'sigm';%'sigm';
    sae.ae{5}.learningRate              = 1;
    sae.ae{5}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{5}.sparsityTarget            = sparsity;
    sae.ae{5}.dropoutFraction           = dropout;
    sae.ae{5}.nonSparsityPenalty        = sparsityPenalty;    
    sae.ae{5}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{6}.activation_function       = 'sigm';%'sigm';
    sae.ae{6}.learningRate              = 1;
    sae.ae{6}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{6}.sparsityTarget            = sparsity;
    sae.ae{6}.dropoutFraction           = dropout;
    sae.ae{6}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{6}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    % back prop
    nn = nnsetup([num_inputs layer1 layer2 layer3 layer4 layer5 layer6 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.W{3} = sae.ae{3}.W{1};
    nn.W{4} = sae.ae{4}.W{1};
    nn.W{5} = sae.ae{5}.W{1};
    nn.W{6} = sae.ae{6}.W{1};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2                  = L2weightPenalty;
    
    % Train the NN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end


end

%% constructAE PCP
% constructs stacked autoencoder
% train_x : training data input (ffts)
% train_y : training data ground truth

function [nn] = constructAEPCP(train_x,train_y,layer1,layer2,layer3,layer4,layer5,layer6,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs,sparsity,dropout,zeroMaskFrac,sparsityPenalty,L2weightPenalty)
disp('constructing SAEPCP')

%truncate data
val_x = train_x(1:10:end,:);
val_y = train_y(1:10:end,:);


train_x(1:10:end,:) = [];
train_y(1:10:end,:) = [];
overflow = mod(size(train_x,1),num_batch);
val_x = [val_x; train_x((end-overflow):end,:)];
val_y = [val_y; train_y((end-overflow):end,:)];
train_x = train_x(1:(end - overflow),:);
train_y = train_y(1:(end - overflow),:);

disp(strcat('training on:',num2str(size(train_x,1)),' samples'));
disp(strcat('validating on:',num2str(size(val_x,1)),' samples'));

%convert y to 1ofK
%convert y to PCP [pcp] = convert1KChordToPCP(K)
disp('converting to pcp and chords')
tic
train_y = convertTo1K(train_y);
val_y = convertTo1K(val_y);
train_y_pcp = convert1KChordToPCPMat(train_y);
val_y_pcp = convert1KChordToPCPMat(val_y);
toc
disp('converted..');

% first train for PCP
if(layer1 > 0 && layer2 == 0)
%train sae
    rand('seed',cputime)
    sae = saesetup([num_inputs layer1]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    
    %options
    opts.numepochs = num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);

    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs layer1 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    nn_pre.sparsityTarget                   = sparsity;
    nn_pre.dropoutFraction                  = dropout;
    nn_pre.nonSparsityPenalty               = sparsityPenalty;
    nn_pre.weightPenaltyL2           = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain_pcp(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs layer1 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;   
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2           = L2weightPenalty;
    % Train the NN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end
if(layer1 > 0 && layer2 > 0 && layer3 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    
    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs layer1 layer2 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    nn_pre.W{2} = sae.ae{2}.W{1};
    nn_pre.sparsityTarget                   = sparsity;
    nn_pre.dropoutFraction                  = dropout;
    nn_pre.nonSparsityPenalty               = sparsityPenalty;
    nn_pre.weightPenaltyL2                  = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain_earlystop(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs layer1 layer2 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    nn.W{3} = nn_pre.W{3};
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2                  = L2weightPenalty;
    % Train the NN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end

if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    
    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs layer1 layer2 layer3 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    nn_pre.W{2} = sae.ae{2}.W{1};
    nn_pre.W{3} = sae.ae{3}.W{1};
    
    nn_pre.sparsityTarget                   = sparsity;
    nn_pre.dropoutFraction                  = dropout;
    nn_pre.nonSparsityPenalty               = sparsityPenalty;
    nn_pre.weightPenaltyL2           = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain_earlystop(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs layer1 layer2 layer3 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    nn.W{3} = nn_pre.W{3};
    nn.W{4} = nn_pre.W{4};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2           = L2weightPenalty;
    % Train the NN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end


if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 > 0 && layer5 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3 layer4]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{4}.activation_function       = 'sigm';%'sigm';
    sae.ae{4}.learningRate              = 1;
    sae.ae{4}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{4}.sparsityTarget            = sparsity;
    sae.ae{4}.dropoutFraction           = dropout;
    sae.ae{4}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{4}.weightPenaltyL2           = L2weightPenalty;
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    
    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs layer1 layer2 layer3 layer4 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    nn_pre.W{2} = sae.ae{2}.W{1};
    nn_pre.W{3} = sae.ae{3}.W{1};
    nn_pre.W{4} = sae.ae{4}.W{1};
    
    nn_pre.sparsityTarget                   = sparsity;
    nn_pre.dropoutFraction                  = dropout;
    nn_pre.nonSparsityPenalty               = sparsityPenalty;
    nn_pre.weightPenaltyL2           = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain_earlystop(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs layer1 layer2 layer3 layer4 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    nn.W{3} = nn_pre.W{3};
    nn.W{4} = nn_pre.W{4};
    nn.W{5} = nn_pre.W{5};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2           = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end

if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 > 0 && layer5 > 0 && layer6 == 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3 layer4 layer5]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{4}.activation_function       = 'sigm';%'sigm';
    sae.ae{4}.learningRate              = 1;
    sae.ae{4}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{4}.sparsityTarget            = sparsity;
    sae.ae{4}.dropoutFraction           = dropout;
    sae.ae{4}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{4}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{5}.activation_function       = 'sigm';%'sigm';
    sae.ae{5}.learningRate              = 1;
    sae.ae{5}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{5}.sparsityTarget            = sparsity;
    sae.ae{5}.dropoutFraction           = dropout;
    sae.ae{5}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{5}.weightPenaltyL2           = L2weightPenalty;
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    
    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs layer1 layer2 layer3 layer4 layer5 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    nn_pre.W{2} = sae.ae{2}.W{1};
    nn_pre.W{3} = sae.ae{3}.W{1};
    nn_pre.W{4} = sae.ae{4}.W{1};
    nn_pre.W{5} = sae.ae{5}.W{1};
        
    nn_pre.sparsityTarget                   = sparsity;
    nn_pre.dropoutFraction                  = dropout;
    nn_pre.nonSparsityPenalty               = sparsityPenalty;
    nn_pre.weightPenaltyL2                  = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain_earlystop(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs layer1 layer2 layer3 layer4 layer5 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    nn.W{3} = nn_pre.W{3};
    nn.W{4} = nn_pre.W{4};
    nn.W{5} = nn_pre.W{5};
    nn.W{6} = nn_pre.W{6};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2                  = L2weightPenalty;    
    
    % Train the NN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end



if(layer1 > 0 && layer2 > 0 && layer3 > 0 && layer4 > 0 && layer5 > 0 && layer6 > 0)

    rand('seed',cputime)
    sae = saesetup([num_inputs layer1 layer2 layer3 layer4 layer5 layer6]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{3}.activation_function       = 'sigm';%'sigm';
    sae.ae{3}.learningRate              = 1;
    sae.ae{3}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{3}.sparsityTarget            = sparsity;
    sae.ae{3}.dropoutFraction           = dropout;
    sae.ae{3}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{3}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{4}.activation_function       = 'sigm';%'sigm';
    sae.ae{4}.learningRate              = 1;
    sae.ae{4}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{4}.sparsityTarget            = sparsity;
    sae.ae{4}.dropoutFraction           = dropout;
    sae.ae{4}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{4}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{5}.activation_function       = 'sigm';%'sigm';
    sae.ae{5}.learningRate              = 1;
    sae.ae{5}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{5}.sparsityTarget            = sparsity;
    sae.ae{5}.dropoutFraction           = dropout;
    sae.ae{5}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{5}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{6}.activation_function       = 'sigm';%'sigm';
    sae.ae{6}.learningRate              = 1;
    sae.ae{6}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{6}.sparsityTarget            = sparsity;
    sae.ae{6}.dropoutFraction           = dropout;
    sae.ae{6}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{6}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    sae = saetrain_isoGauss(sae, train_x, opts);
    
    
    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs layer1 layer2 layer3 layer4 layer5 layer6 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    nn_pre.W{2} = sae.ae{2}.W{1};
    nn_pre.W{3} = sae.ae{3}.W{1};
    nn_pre.W{4} = sae.ae{4}.W{1};
    nn_pre.W{5} = sae.ae{5}.W{1};
    nn_pre.W{5} = sae.ae{5}.W{1};
        
    nn_pre.sparsityTarget                   = sparsity;
    nn_pre.dropoutFraction                  = dropout;
    nn_pre.nonSparsityPenalty               = sparsityPenalty;
    nn_pre.weightPenaltyL2                  = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain_earlystop(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs layer1 layer2 layer3 layer4 layer5 layer6 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    nn.W{3} = nn_pre.W{3};
    nn.W{4} = nn_pre.W{4};
    nn.W{5} = nn_pre.W{5};
    nn.W{6} = nn_pre.W{6};
    nn.W{7} = nn_pre.W{7};
    
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2           = L2weightPenalty;
    
    % Train the NN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end

end


%% countHMMParams(gts_full)
% gets ground truth, computes the parameters of the HMM in a supervised
% manner
% gts_full : cell array of matrices(1,num_frames) for each song in training
% set
% outputs:
% inits : initial state probability of the hmm
% transitions : transition probability for hmm

function [inits,transitions] = countHMMParams(gts_full)
transitions = zeros(25,25);
inits = zeros(25);

%every song
for song = 1:length(gts_full)
    s = gts_full{song};
    % every frame in the song, everything is one dimensional, so all good
    previous = s(1) + 1; % add one for correct index for chord (matlab sigh)
    inits(previous) = inits(previous) + 1;
    for frame = 2:length(s)
        now = s(frame) + 1;
        transitions(previous,now) = transitions(previous,now) + 1;
        previous = now;
    end
end

% no ones left behind:
transitions = transitions + ones(size(transitions));
inits = inits + ones(size(inits));

% normalize transition TODO:carefull i'm tired, might not work correctly.
X = transitions';
X = X./( ones(size(X)) * diag(sum(abs(X))) );
transitions = X';
%normalize initial probability distribution
inits = inits / sum(inits);
end


%% constructDBN 
% constructs stacked autoencoder
% train_x : training data input (ffts)
% train_y : training data ground truth

function [nn] = constructDBN(train_x,train_y,firstlayer,secondlayer)
disp('constructing DBN')
% parameters
num_batch = 100;

%truncate data

val_x = train_x(1:10:end,:);
val_y = train_y(1:10:end,:);


train_x(1:10:end,:) = [];
train_y(1:10:end,:) = [];
overflow = mod(size(train_x,1),num_batch);
val_x = [val_x; train_x((end-overflow):end,:)];
val_y = [val_y; train_y((end-overflow):end,:)];
train_x = train_x(1:(end - overflow),:);
train_y = train_y(1:(end - overflow),:);


%convert y in 1ofK
train_y = convertTo1K(train_y);
val_y = convertTo1K(val_y);
if(firstlayer > 0 && secondlayer == 0)
%train dbn
%%  ex1 train a 100 hidden unit RBM and visualize its weights
    rand('seed',cputime)
    dbn.sizes = [100];
    opts.numepochs =   30;
    opts.batchsize = num_batch;
    opts.momentum  =   0;
    opts.alpha     =   1;
    dbn = dbnsetup(dbn, train_x, opts);
    dbn = dbntrain(dbn, train_x, opts);
    figure; visualize(dbn.rbm{1}.W);   %  Visualize the RBM weights
    %back prop
    %unfold dbn to nn
    nn = dbnunfoldtonn(dbn, 25);

    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    
    % Train the FFNN
    opts.numepochs =  50;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop(nn, train_x, train_y, opts, val_x, val_y);
end
end
