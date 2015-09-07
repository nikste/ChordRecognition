function [nn] = initTrain7Inv(trainfiles,gtfiles)
    zeroMaskFrac = 0.7;
    dropout = 0.5;
    sparsity = 0.05;
    sparsityPenalty = 0.1;
    vari = 0.2;
    L2weightPenalty = 0;
    num_inputs = 1500;
    layer1 = 800;
    layer2 = 800;
    num_train_unsup = 30;
    num_batch = 1000;
    
    num_train_chord = 50;
    
    
    sae = saesetup([num_inputs layer1 layer2]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.momentum                  = 0.5;
    sae.ae{1}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = dropout;
    sae.ae{1}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{1}.weightPenaltyL2           = L2weightPenalty;
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.momentum                  = 0.5;
    sae.ae{2}.inputZeroMaskedFraction   = zeroMaskFrac;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = dropout;
    sae.ae{2}.nonSparsityPenalty        = sparsityPenalty;
    sae.ae{2}.weightPenaltyL2           = L2weightPenalty;
    
    %options
    opts.numepochs = num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    opts.noisevariance = vari;
    %sae = saetrain_isoGauss(sae, train_x, opts);
    
    %%sae = saetrain_isoGauss27Inv(sae,opts, trainfiles)
    %%save('C:\stuff\masterthesis\sae','sae');
    % back prop
    nn = nnsetup([num_inputs layer1 layer2 217]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.sparsityTarget                   = sparsity;
    nn.dropoutFraction                  = dropout;
    nn.nonSparsityPenalty               = sparsityPenalty;
    nn.weightPenaltyL2                  = L2weightPenalty;
        
    % Train the NN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_earlystop27inv(nn, opts,trainfiles,gtfiles);
    save('C:\stuff\masterthesis\nn','nn');
end