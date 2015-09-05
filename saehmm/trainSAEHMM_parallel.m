function [nn_pcp,nn,inits,transitions]= trainSAEHMM_parallel(ffts_train,gts_train,gts_full)
%[sae,nn,im1,im2,im3,err,inits,transitions] =

%% initialize HMM
disp('initializing hmm')
[inits,transitions] = countHMMParams(gts_full);


%% train sae on shrunk dataset
% for safety reasons:
assert(size(gts_train,1) == size(ffts_train,1));


% does preprocessing internally
% preprocess data
%[ffts_train_,M,P] = preprocessData(ffts_train);
%ffts_train = sqrt(ffts_train);
%ffts_train = normr(ffts_train);

num_batch = 2;
num_train_unsup = 5;
num_train_pcp = 30;
num_train_chord = 20;
firstlayer = 200;
secondlayer = 0;
funcs = {@constructAE,@constructAEPCP};
arguments = {ffts_train,gts_train,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord;
    ffts_train,gts_train,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord};
solutions = cell(1,2);

parfor i = 1:2
        %nn = constructAE(ffts_train,gts_train,hidden_layer1,hidden_layer2,num_train_unsup,num_train_chord);
        solutions{i}=funcs{i}(arguments{i,:});
        %nn_pcp = constructAEPCP(ffts_train,gts_train,hidden_layer1,hidden_layer2,num_train_unusp,num_train_pcp,num_train_chord);%constructAE(ffts_train,gts_train,200,0); %constructAE(ffts_train,gts_train,200,0);    end
end
%% return trained machine.
disp('enjoy your result!')
%done
nn = solutions{1};
nn_pcp = solutions{2};
end


%% constructAE 
% constructs stacked autoencoder
% train_x : training data input (ffts)
% train_y : training data ground truth

function [nn] = constructAE(train_x,train_y,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord)
disp('constructing SAE')
% parameters

%truncate data
num_inputs = 1500;
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
%train sae
    rand('state',0)
    sae = saesetup([num_inputs firstlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.7;
    
    %options
    opts.numepochs = num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);
%back prop
    nn = nnsetup([num_inputs firstlayer 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    
    % Train the FFNN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
  end
if(firstlayer > 0 && secondlayer > 0)

    rand('state',0)
    sae = saesetup([num_inputs firstlayer secondlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.7;

        sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = 0.7;
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);
%back prop
    nn = nnsetup([num_inputs firstlayer secondlayer 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    
    % Train the FFNN
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);


end
end

%% constructAE 
% constructs stacked autoencoder
% train_x : training data input (ffts)
% train_y : training data ground truth

function [nn] = constructAEPCP(train_x,train_y,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord)
disp('constructing SAEPCP')

%truncate data
num_inputs = 1500;
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
if(firstlayer > 0 && secondlayer == 0)
%train sae
    rand('state',0)
    sae = saesetup([num_inputs firstlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.7;
    
    %options
    opts.numepochs = num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);

    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs firstlayer 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    
    % Train the FFNN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain_pcp(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs firstlayer 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    
    % Train the FFNN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
    
    
    
    
  end
if(firstlayer > 0 && secondlayer > 0)

    rand('state',0)
    sae = saesetup([num_inputs firstlayer secondlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.7;

        sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = 0.7;
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);
    
    
    
    %back prop for PCP
    nn_pre = nnsetup([num_inputs firstlayer secondlayer 12]);
    
    %options
    nn_pre.activation_function              = 'sigm';%'sigm';
    nn_pre.output                           = 'softmax';
    nn_pre.learningRate                     = 1;
    nn_pre.W{1} = sae.ae{1}.W{1};
    nn_pre.W{2} = sae.ae{2}.W{1};
    
    % Train the FFNN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn_pre = nntrain(nn_pre, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs firstlayer secondlayer 12 25]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = nn_pre.W{1};
    nn.W{2} = nn_pre.W{2};
    nn.W{3} = nn_pre.W{3};
    
    % Train the FFNN for PCP
    opts.numepochs =  num_train_chord;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
    
    
%back prop
%     nn = nnsetup([num_inputs firstlayer secondlayer 12]);
%     
%     %options
%     nn.activation_function              = 'sigm';%'sigm';
%     nn.output                           = 'softmax';
%     nn.learningRate                     = 1;
%     nn.W{1} = sae.ae{1}.W{1};
%     nn.W{2} = sae.ae{2}.W{1};
%     
%     % Train the FFNN
%     opts.numepochs =  50;
%     opts.batchsize = num_batch;%6203;%2189; %308;%100;
%     opts.plot = 1;
%     nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);


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

