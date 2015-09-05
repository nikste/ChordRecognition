function [nn,inits,transitions, mus,sigmas,mixmats] = trainJointPre(chords_sorted_fft,gts_full,firstlayer,secondlayer, multi_res, pcp_pretrain)
%TRAINJOINTPRE Summary of this function goes here
%   Detailed explanation goes here

%% construct training data for neural network
disp('constructing train data');
l = 0;
for c=1:length(chords_sorted_fft)
   l = l + size(chords_sorted_fft{c},1); 
end
gts_train = zeros(l,1);
ctr = 1;
for c=1:length(chords_sorted_fft)
    this_length = size(chords_sorted_fft{c},1);
    gts_train(ctr:(ctr+this_length-1),1) = c - 1; %cause chords
    ctr = ctr + this_length;

end

ffts_train = [];
for c=1:length(chords_sorted_fft)
   ffts_train = [ffts_train; chords_sorted_fft{c}]; 
end
size(gts_train)
size(ffts_train)
%ffts_train = cell2mat(chords_sorted_fft);


%% neural network training:
disp(strcat('training neural network on:',num2str(l),' datapoints'));
num_inputs=1500;
num_batch = 100;
disp(strcat('num_batch:',num2str(num_batch)));
num_train_unsup = 10;%1
disp(strcat('num_train_unsup:',num2str(num_train_unsup)));
num_train_pcp = 100;%1
disp(strcat('num_train_pcp',num2str(num_train_pcp)));
num_train_chord = 100;%1
disp(strcat('num_train_chord',num2str(num_train_chord)));
r = randperm(size(ffts_train,1));
ffts_train = ffts_train(r,:);
gts_train = gts_train(r,:);
[nn] = constructAEPCP(ffts_train,gts_train,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs);

%% compute features
disp('recomputing features')

[pcps] = runNN(nn,ffts_train);
% pcps = rand(size(ffts_train,1),12);
% disp(strcat('pcps:',num2str(size(pcps))))
% disp(strcat('gst_train:',num2str(size(gts_train))))
[chords_sorted_pcp] = sortChords( pcps, gts_train );

ffts_train = [];
%% gaussian mixture hmm
disp('training hmm on neural network output')
number_components = 2 ;
disp(strcat('number of components:',num2str(number_components)));
[inits,transitions, mus,sigmas,mixmats ] = train_gmms( chords_sorted_pcp ,gts_full, number_components);

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
    nn = nnsetup([num_inputs firstlayer 12]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';%'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.dropoutFraction = 0.5;
    % Train the FFNN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain_pcp(nn, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
%     %back prop for PCP
%     nn = nnsetup([num_inputs firstlayer 12 25]);
%     
%     %options
%     nn.activation_function              = 'sigm';%'sigm';
%     nn.output                           = 'softmax';
%     nn.learningRate                     = 1;
%     nn.W{1} = nn_pre.W{1};
%     nn.W{2} = nn_pre.W{2};
%     
%     % Train the FFNN for PCP
%     opts.numepochs =  num_train_chord;
%     opts.batchsize = num_batch;%6203;%2189; %308;%100;
%     opts.plot = 1;
%     nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
    
    
    
    
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
    sae.dropoutFraction = 0.5;
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);
    
    
    
    %back prop for PCP
    nn = nnsetup([num_inputs firstlayer secondlayer 12]);
    
    %options
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';%'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.dropoutFraction = 0.5;
    % Train the FFNN for PCP
    opts.numepochs =  num_train_pcp;
    opts.batchsize = num_batch;%6203;%2189; %308;%100;
    opts.plot = 1;

    nn = nntrain(nn, train_x, train_y_pcp, opts, val_x, val_y_pcp);
    
    
%     %back prop for PCP
%     nn = nnsetup([num_inputs firstlayer secondlayer 12 25]);
% 
%     %options
%     nn.activation_function              = 'sigm';%'sigm';
%     nn.output                           = 'softmax';
%     nn.learningRate                     = 1;
%     
%     nn.W{1} = nn_pre.W{1};
%     nn.W{2} = nn_pre.W{2};
%     nn.W{3} = nn_pre.W{3};
%     
%     % Train the FFNN for PCP
%     opts.numepochs =  num_train_chord;
%     opts.batchsize = num_batch;%6203;%2189; %308;%100;
%     opts.plot = 1;
%     nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
    
    
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