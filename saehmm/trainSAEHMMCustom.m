function [nn_left,nn_middle,nn_right,nn_top,inits,transitions]= trainSAEHMMCustom(ffts_train,gts_train,gts_full,first_layer,second_layer, multi_res, pcp_pretrain)
%[sae,nn,im1,im2,im3,err,inits,transitions] =

%% initialize HMM
disp('initializing hmm')
[inits,transitions] = countHMMParams(gts_full);

disp(strcat('size of ground truths:',num2str(size(gts_train,1))));
disp(strcat('size of ffts:',num2str(size(ffts_train,1))));
%% train sae on shrunk dataset
% for safety reasons:
assert(size(gts_train,1) == size(ffts_train,1),'length of ffts and ground truth dont match!');

num_inputs = 5;
num_batch = 100;
disp(strcat('num_batch:',num2str(num_batch)));
num_train_unsup = 5;
disp(strcat('num_train_unsup:',num2str(num_train_unsup)));
num_train_pcp = 5;
disp(strcat('num_train_pcp',num2str(num_train_pcp)));
num_train_chord = 50;
disp(strcat('num_train_chord',num2str(num_train_chord)));

%randomize training samples
r = randperm(size(ffts_train,1));
ffts_train = ffts_train(r,:);
gts_train = gts_train(r,:);

[nn_left,nn_middle,nn_right,nn_top] = constructAECustom(ffts_train,gts_train,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs)
% if (multi_res == 1)
%     if (pcp_pretrain == 1)
%         nn = constructAE(ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,4500);
%     else
%         nn = constructAE(ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,4500);%constructAE(ffts_train,gts_train,200,0); %constructAE(ffts_train,gts_train,200,0);
%     end
% else
%     if (pcp_pretrain == 1)
%         nn = constructAEPCP(ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,1500);
%     else
%         nn = constructAE(ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,1500);
%     end
% end
%% return trained machine.
disp('enjoy your result!')
%done


end

function [nn_left,nn_middle,nn_right,nn_top] = constructAECustom(train_x,train_y,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs)
disp('constructing custom SAE')
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



% trains Network
%   
%      25 
%       |
%      200
%    /  |  \
% 200  200  200
%  |    |    |
%1500  1500 1500


% train first layer nn:

%train sae_left
rand('state',0)
sae_left = saesetup([1500 200]);
sae_left.ae{1}.activation_function       = 'sigm';%'sigm';
sae_left.ae{1}.learningRate              = 1;
sae_left.ae{1}.inputZeroMaskedFraction   = 0.7;

%options
opts.numepochs = num_train_unsup;
opts.batchsize = num_batch;%74436;%2189;%100
sae_left = saetrain(sae_left,  train_x(1:end,1:1500), opts);


% train second nn:

%train sae_middle
rand('state',0)
sae_middle = saesetup([1500 200]);
sae_middle.ae{1}.activation_function       = 'sigm';%'sigm';
sae_middle.ae{1}.learningRate              = 1;
sae_middle.ae{1}.inputZeroMaskedFraction   = 0.7;

%options
opts.numepochs = num_train_unsup;
opts.batchsize = num_batch;%74436;%2189;%100
sae_middle = saetrain(sae_middle, train_x(1:end,1501:3000), opts);


% train thid nn:

%train sae_middle
rand('state',0)
sae_right = saesetup([1500 200]);
sae_right.ae{1}.activation_function       = 'sigm';%'sigm';
sae_right.ae{1}.learningRate              = 1;
sae_right.ae{1}.inputZeroMaskedFraction   = 0.7;

%options
opts.numepochs = num_train_unsup;
opts.batchsize = num_batch;%74436;%2189;%100
sae_right = saetrain(sae_right, train_x(1:end,3001:4500), opts);

train_x_current = [];

% convert to neural networks:
    nn_left = nnsetup([1500 200]);
    
    %options
    nn_left.activation_function              = 'sigm';%'sigm';
    nn_left.output                           = 'sigm';
    nn_left.learningRate                     = 1;
    nn_left.W{1} = sae_left.ae{1}.W{1};
    nn_left.dropoutFraction = 0.5;
    
    % Train the FFNN
    %opts.numepochs =  num_train_chord;
    %opts.batchsize = num_batch;%6203;%2189; %308;%100;
    %opts.plot = 1;
    %nn_left = nntrain(nn_left, train_x, train_y, opts, val_x, val_y);
    nn_middle = nnsetup([1500 200]);
    
    %options
    nn_middle.activation_function              = 'sigm';%'sigm';
    nn_middle.output                           = 'sigm';
    nn_middle.learningRate                     = 1;
    nn_middle.W{1} = sae_middle.ae{1}.W{1};
    nn_middle.dropoutFraction = 0.5;
    
    
    nn_right = nnsetup([1500 200]);
    
    %options
    nn_right.activation_function              = 'sigm';%'sigm';
    nn_right.output                           = 'sigm';
    nn_right.learningRate                     = 1;
    nn_right.W{1} = sae_right.ae{1}.W{1};
    nn_right.dropoutFraction = 0.5;
    
% propagate whole dataset
left = runNN(nn_left,train_x(1:end,1:1500));
middle = runNN(nn_middle,train_x(1:end,1501:3000));
right = runNN(nn_right,train_x(1:end,3001:4500));

train_up = [left middle right];
left = [];
middle = [];
right = [];

% train sae up
%train sae_middle
rand('state',0)
sae_top = saesetup([600 200]);
sae_top.ae{1}.activation_function       = 'sigm';%'sigm';
sae_top.ae{1}.learningRate              = 1;
sae_top.ae{1}.inputZeroMaskedFraction   = 0.7;

%options
opts.numepochs = num_train_unsup;
opts.batchsize = num_batch;%74436;%2189;%100
sae_top = saetrain(sae_top, train_up, opts);


% stacked autoencoders 
%   initialized!
%   
%      25 
%       |
%      200
%    /  |  \
% 200  200  200
%  |    |    |
%1500  1500 1500

%% Backpropagation
    nn_top = nnsetup([600 200 25]);
    
    %options
    nn_top.activation_function              = 'sigm';%'sigm';
    nn_top.output                           = 'softmax';
    nn_top.learningRate                     = 1;
    nn_top.W{1} = sae_top.ae{1}.W{1};
    nn_top.dropoutFraction = 0.5;
    
    
    
    


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

%% constructAE 
% constructs stacked autoencoder
% train_x : training data input (ffts)
% train_y : training data ground truth

function [nn] = constructAE(train_x,train_y,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs)
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
    nn.dropoutFraction = 0.5;
    
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
    nn.dropoutFraction = 0.5;
    
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

function [nn] = constructAEPCP(train_x,train_y,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs)
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
    nn_pre.dropoutFraction = 0.5;
    
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
    nn.dropoutFraction = 0.5;    
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
    nn_pre.dropoutFraction = 0.5;
    
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
    nn.dropoutFraction = 0.5;
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
    rand('state',0)
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
    nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
  end
% if(firstlayer > 0 && secondlayer > 0)
% 
%     rand('state',0)
%     sae = saesetup([1500 firstlayer secondlayer]);
%     sae.ae{1}.activation_function       = 'sigm';%'sigm';
%     sae.ae{1}.learningRate              = 1;
%     sae.ae{1}.inputZeroMaskedFraction   = 0.7;
% 
%         sae.ae{2}.activation_function       = 'sigm';%'sigm';
%     sae.ae{2}.learningRate              = 1;
%     sae.ae{2}.inputZeroMaskedFraction   = 0.7;
%     %options
%     opts.numepochs =  50;
%     opts.batchsize = num_batch;%74436;%2189;%100
%     sae = saetrain(sae, train_x, opts);
% %back prop
%     nn = nnsetup([1500 firstlayer secondlayer 25]);
%     
%     %options
%     nn.activation_function              = 'sigm';%'sigm';
%     nn.output                           = 'softmax';
%     nn.learningRate                     = 1;
%     nn.W{1} = sae.ae{1}.W{1};
%     nn.W{2} = sae.ae{2}.W{1};
%     
%     % Train the FFNN
%     opts.numepochs =  200;
%     opts.batchsize = num_batch;%6203;%2189; %308;%100;
%     opts.plot = 1;
%     nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
% 
% 
% end
end


