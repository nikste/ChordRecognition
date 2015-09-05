function [sae] = trainSAEHMMPre(sae1,ffts_train,gts_train,gts_full,first_layer,second_layer,multi_res,pcp_pretrain)
% does pretraining to speed up grid search.


%% train sae on shrunk dataset
% for safety reasons:
assert(size(gts_train,1) == size(ffts_train,1),'length of ffts and ground truth dont match!');

disp(strcat('firstlayer:',num2str(first_layer)));
disp(strcat('secondlayer:',num2str(second_layer)));

num_batch = 100;
disp(strcat('num_batch:',num2str(num_batch)));
num_train_unsup = 5;
disp(strcat('num_train_unsup:',num2str(num_train_unsup)));
num_train_pcp = 100;
disp(strcat('num_train_pcp',num2str(num_train_pcp)));
num_train_chord = 100;
disp(strcat('num_train_chord',num2str(num_train_chord)));

sparsity = 0.05;
disp(strcat('sparsity_target',num2str(sparsity)));

%randomize training samples
r = randperm(size(ffts_train,1));
ffts_train = ffts_train(r,:);
gts_train = gts_train(r,:);


% if second layer not empty use sae weights supplied in the function,
% otherwise dont.
% 
% if(first_layer > 0 && second_layer > 0)
    %use sae supplied 
    if (multi_res == 1)
        sae = constructAE(sae1,ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,4500,sparsity);%constructAE(ffts_train,gts_train,200,0); %constructAE(ffts_train,gts_train,200,0);
    else
        sae = constructAE(sae1,ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,1500,sparsity);
    end 
    
% elseif(first_layer > 0 && second_layer == 0)
%     if (multi_res == 1)
%         sae = constructAEFirst(sae1,ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,4500,sparsity);%constructAE(ffts_train,gts_train,200,0); %constructAE(ffts_train,gts_train,200,0);
%     else
%         sae = constructAEFirst(sae1,ffts_train,gts_train,first_layer,second_layer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,1500,sparsity);
%     end
% else
%     disp('you suck, something is wrong with the layer constellation');
% end

%% return trained machine.
disp('enjoy your result!')
%done


end


%% constructAE 
% constructs stacked autoencoder
% train_x : training data input (ffts)
% train_y : training data ground truth

function [sae] = constructAE(sae1,train_x,train_y,firstlayer,secondlayer,num_batch,num_train_unsup,num_train_pcp,num_train_chord,num_inputs,sparsity)
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
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = 0.5;
    
    %options
    opts.numepochs = num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);
end
% if we want to use a second layer, we'd better supply a pretraining for
% first.
if(firstlayer > 0 && secondlayer > 0)

    rand('state',0)
    sae = saesetup([num_inputs firstlayer secondlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.7;
    sae.ae{1}.sparsityTarget            = sparsity;
    sae.ae{1}.dropoutFraction           = 0.5;
    sae.ae{1}.W{1}                      = sae1.ae{1}.W{1};
    
    
    sae.ae{2}.activation_function       = 'sigm';%'sigm';
    sae.ae{2}.learningRate              = 1;
    sae.ae{2}.inputZeroMaskedFraction   = 0.7;
    sae.ae{2}.sparsityTarget            = sparsity;
    sae.ae{2}.dropoutFraction           = 0.5;
    %options
    opts.numepochs =  num_train_unsup;
    opts.batchsize = num_batch;%74436;%2189;%100
    %sae = saetrain(sae, train_x, opts);
    
    % copied from saetrain, to skip first layer
    
    %prepare training data for second layer
    t = nnff(sae.ae{1}, train_x, train_x);
    x = t.a{2};
    % remove bias term
    x = x(:,2:end);
    
    % train second layer:
    i=2;
    disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
    sae.ae{i} = nntrain(sae.ae{i}, x, x, opts);
    
    end
end






