function [nn, L]  = nntrain_isoGauss27Inv(sae, opts, layer, filelist)%nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

%assert(isfloat(train_x), 'train_x must be a float');
%assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')


%opts.numepochs = 30;


loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
%if nargin == 6
%    opts.validation = 1;
%end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end


batchsize = opts.batchsize;
numepochs = opts.numepochs;


%% TOOD: pull this out
partitions = 4;

n = 1;

nn = sae.ae{layer};
%filelist = loadFilenames(id);

fhandle = figure();
opts.plot = 1;
opts.validation = 1;


nn.inputZeroMaskedFraction = 0.7;
nn.dropoutFraction = 0.7;
nn.sparsityTarget = 0.05;
nn.nonSparsityPenalty = 0.1;
nn.dropoutFraction = 0.5;
i_inc = 0;
for p = 1 : partitions
    tic;

    %loadfiles
    filelist_local = filelist(p : partitions : end); 
    
    train_x = loadFiles(filelist_local);
    %preprocess -> is done in load function
    %cut of too many frames if there are.
    
    
    if(layer > 1)
        for j = 1:layer - 1
            t = nnff(sae.ae{j}, train_x, train_x);
            train_x = t.a{2};
            %remove bias term
            train_x = train_x(:,2:end);
        end
    end
    last_point = size(train_x,1) - mod(size(train_x,1),batchsize);
    val_x = train_x(last_point : end,:);
    val_y = val_x;
    train_x = train_x(1:last_point,:);
    train_y = train_x;
    
    
    
    t_load = toc;
    disp(['partition ' num2str(p) ' / ' num2str(partitions) ' loading took ' num2str(t_load) ' seconds']);
      
    m = size(train_x, 1);
    numbatches = m / batchsize;            
    assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');
    L = zeros(numepochs*numbatches,1);
    
    for i = 1 : numepochs
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);

            %Add noise to input (for use in denoising autoencoder)
            if(nn.inputZeroMaskedFraction ~= 0)
                %% set only to zero if smaller than inputZeroMaskedFraction.
                %% batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
                %% isotropic gaussian noise
                mak = (rand(size(batch_x)) < nn.inputZeroMaskedFraction);
                noise = ( opts.noisevariance .* randn(size(batch_x))  ) .* mak; % wtf chol? chol(opts.nosievariance)
                batch_x = batch_x + noise;
            end

            batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);

            nn = nnff(nn, batch_x, batch_y);
            nn = nnbp(nn);
            nn = nnapplygrads(nn);

            L(n) = nn.L;

            n = n + 1;
        end

        t = toc;

        if opts.validation == 1
            loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
            str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
        else
            loss = nneval(nn, loss, train_x, train_y);
            str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
        end
        if ishandle(fhandle)
            i_inc = i_inc + 1;
            %save_epochs = opts.numepochs;
            %opts.numepochs = i;
            nnupdatefigures_incremental(nn,fhandle,loss,opts,i_inc)
            %nnupdatefigures(nn, fhandle, loss, opts, i);
            %opts.numepochs = save_epochs;
        end

        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
        nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    end
end
disp(['finished']);
end


%% not used at the moment.
function filenames = loadFilenames(id)
    filenames = {};
    %% load blacklist,
    blacklistfile = strcat('C:\stuff\masterthesis\blacklist_',num2str(id));
    blacklist = importdata(blacklistfile);
    % load all filenames from gt folder (this does not need to be optimized)
    gtfolder = 'C:\stuff\masterthesis\gt';
    gtfiles_all = dir(strcat(gtfolder,'\*.dataC'));

    trainfiles = {};
    blacklist2 = {};
    %convert to strings
    blacklist2 = convertToStrings(blacklist);
    blacklist = blacklist2';

    trainfilectr = 1;
    for ind = 1:length(gtfiles_all)
        [pathstr,name,ext] = fileparts(gtfiles_all(ind).name);
        if(ismember(name,blacklist) == 0)
            filenames{trainfilectr} = name;
            trainfilectr = trainfilectr + 1;
        else
            disp(strcat('omitted:',name));
        end
    end
end


function [train_x,train_y] = loadFiles(filelist_local)
%fftfoldername = 'C:\stuff\masterthesis\data\fft';
train_x = [];
for ind = 1 : 8:  size(filelist_local,2)
    %filename = strcat(fftfoldername,'\',filelist_local{ind},'.dataF');
    %disp(['loading file:' filename ])
    filename = filelist_local{ind};
    train_x_ = load(filename,'-mat');
    train_x_ = train_x_.data;
    train_x = [train_x;train_x_];
end
end


