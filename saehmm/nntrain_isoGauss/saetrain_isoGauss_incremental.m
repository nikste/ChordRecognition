function [nn, L]  = saetrain_isoGauss_incremental(sae,train_layer_num, opts, id, shrinkfactor, is_unsupervised)
%NNTRAIN trains a neural net nntrain
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

nn = sae.ae{train_layer_num};

trainfilelist = get_train_files(id);
%assert(isfloat(train_x), 'train_x must be a float');
%assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
fhandle = figure();
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

%m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

%numbatches = m / batchsize;

%assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

%L = zeros(numepochs*numbatches,1);
%n = 1;
jobs_cell = {};
figure_num_epochs_counter = 0;
for i = 1 : numepochs
    
    
    trainfilelist_local = trainfilelist(1:shrinkfactor:end);
    jobs_cell{1} = start_jobs(trainfilelist_local);
    trainfilelist_local = trainfilelist(2:shrinkfactor:end);
    jobs_cell{2} = start_jobs(trainfilelist_local);
    
    %[train_x,train_y,val_x,val_y] = get_more_train(trainfilelist_local,is_unsupervised,batchsize);
    for offset = 0 : shrinkfactor - 1
        tic
        
        % load part of dataset
        trainfilelist_local = trainfilelist(offset+1:shrinkfactor:end);
        jobs = jobs_cell{mod(offset,2)+1};
        [train_x,train_y,val_x,val_y] = stop_jobs(jobs,trainfilelist_local,batchsize);
       
        %restart job with new files
        trainfilelist_local = trainfilelist(mod(offset+2,shrinkfactor)+1:shrinkfactor:end);
        jobs_cell{mod(offset,2)+1} = start_jobs(trainfilelist_local);
         
        
        if(is_unsupervised)
            train_y = train_x;
            val_y = val_x;
        end
        
        % if its next layer, propagate accordingly.
        if(train_layer_num > 1)
            for tln = 1:train_layer_num - 1
                t = nnff(sae.ae{tln}, train_x, train_y);
                train_x = t.a{2};
                %remove bias term
                train_x = train_x(:,2:end);
                if(is_unsupervised)
                    train_y = train_x;
                end
            end
        end
        if(train_layer_num > 1)
            for tln = 1:train_layer_num - 1
                t = nnff(sae.ae{tln}, val_x, val_y);
                val_x = t.a{2};
                %remove bias term
                val_x = val_x(:,2:end);
                if(is_unsupervised)
                    val_y = val_x;
                end
            end
        end

        loading_t = toc;
        tic
        
        
        m = size(train_x, 1);
        numbatches = m / batchsize;

        assert(rem(numbatches, 1) == 0)%, 'numbatches must be a integer');

        L = zeros(numepochs*numbatches,1);
        n = 1;

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
                batch_x = batch_x +  noise;
            end

            batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);

            nn = nnff(nn, batch_x, batch_y);
            nn = nnbp(nn);
            nn = nnapplygrads(nn);

            L(n) = nn.L;

            n = n + 1;
        end

        
        figure_num_epochs_counter = figure_num_epochs_counter + 1;
        if opts.validation == 1
            loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
            str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
        else
            loss = nneval(nn, loss, train_x, train_y);
            str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
        end
        if ishandle(fhandle)
            nnupdatefigures_incremental(nn, fhandle, loss, opts, figure_num_epochs_counter);
        end
                
        
    
        
        disp('job finished')
        t = toc;
        disp(['epoch ' num2str(offset) '/' num2str(shrinkfactor) '|' num2str(i) '/' num2str(opts.numepochs) '. training ' num2str(t) ' loading ' num2str(loading_t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
        
        %[train_x,train_y,val_x,val_y] = fetchOutputs(f); % Blocks until complete
        nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    end
    
        
end
terminate_all_jobs(jobs_cell);
delete('C:\Users\nist03\AppData\Roaming\MathWorks\MATLAB\local_cluster_jobs\R2013b\*');
rmdir('C:\Users\nist03\AppData\Roaming\MathWorks\MATLAB\local_cluster_jobs\R2013b\*','s');
end

function [] = terminate_all_jobs(jobs_cell)
disp('finished, cancelling jobs!')
for j = 1:length(jobs_cell)
   for k = 1:length(jobs_cell{j})
   cancel(jobs_cell{j}{k});
   end
end
end



function jobs = start_jobs(trainfilelist_local)

    jobs = cell(length(trainfilelist_local));
    for job_no = 1:length(trainfilelist_local)
       jobs{job_no} =  batch(@load_trainfiles_local_parallel,2,{trainfilelist_local{job_no}});
    end
    
end

function [train_x,train_y,val_x,val_y] = stop_jobs(jobs,trainfilelist_local,batchsize)
train_x = cell(length(trainfilelist_local));
train_y = cell(length(trainfilelist_local));
val_x = [];
val_y = [];
for job_no = 1:length(trainfilelist_local)
    wait(jobs{job_no})
end

for job_no = 1:length(trainfilelist_local)
    res = fetchOutputs(jobs{job_no});
    train_x{job_no} = res{1};
    train_y{job_no} = res{2};
end
train_x = cell2mat(train_x);
train_y = cell2mat(train_y);

for job_no = 1:length(trainfilelist_local)
   delete(jobs{job_no}) 
end
[train_x,train_y,val_x,val_y] = split_train_and_validation(train_x,train_y,batchsize);

end




function [train_x,train_y,val_x,val_y] = get_more_train(trainfilelist_local,is_unsupervised,batchsize)

   trainfilelist_local
    disp(is_unsupervised)
    disp(batchsize)
    [train_x,train_y] = load_trainfiles_local(trainfilelist_local);
    [train_x,train_y,val_x,val_y] = split_train_and_validation(train_x,train_y,batchsize);
    
    if(is_unsupervised)
        train_y = train_x;
        val_y = val_x;
    end
end


function trainfilelist = get_train_files(id)

%% load blacklist,
base_dir = 'C:\stuff\masterthesis\';
base_dir_linux = '/media/nikste/moarspace/masterthesis/';
blacklistfile = strcat(base_dir,'blacklist_',num2str(id));
blacklist = importdata(blacklistfile);

%% load all files not in blacklist

% load all filenames from gt folder (this does not need to be optimized)
gtfolder = strcat(base_dir,'gt');%'/media/nikste/moarspace/masterthesis/gt';

gtfiles_all = dir([gtfolder,'\*.dataC']);%myls(strcat(gtfolder,'\*.dataC'));

trainfiles = {};
blacklist2 = {};
%convert to strings
blacklist2 = convertToStrings(blacklist);
blacklist = blacklist2';

trainfilectr = 1;
for ind = 1:length(gtfiles_all)
    [pathstr,name,ext] = fileparts(gtfiles_all(ind).name);
    if(ismember(name,blacklist) == 0)
        trainfiles{trainfilectr} = name;
        trainfilectr = trainfilectr + 1;
    end
end


trainfilelist = trainfiles;
end


function [train_x,train_y,val_x,val_y] = split_train_and_validation(train_x,train_y,num_batch)
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
end


%%
% loads one file (for parallel loading)
function [ffts_loaded,gts_loaded] = load_trainfiles_local_parallel(trainfile_local)
    base_dir = 'C:\stuff\masterthesis\'
    fftfoldername = strcat(base_dir,'data\fft');%'/media/nikste/moarspace/masterthesis/data/fft';
    gtfoldername = strcat(base_dir,'gt');%'/media/nikste/moarspace/masterthesis/gt';
    ['loading files:' fftfoldername 'and' gtfoldername]
     %ffts_loaded = importdata(strcat(fftfoldername,'/',trainfile_local,'.dataF'));
     ffts_loaded = load(strcat(fftfoldername,'\',trainfile_local,'.dataF'),'-mat');
     ffts_loaded = ffts_loaded.data
     ffts_loaded = normr(sqrt(ffts_loaded));
     gts_loaded = importdata(strcat(gtfoldername,'\',trainfile_local,'.dataC'));
     gts_loaded = gts_loaded(:,1);

     d = size(gts_loaded,1) - size(ffts_loaded,1);
    if(d ~= 0)
        if(d > 0) % add frame
            for i=1:d
                ffts_loaded = [ffts_loaded;ffts_loaded(end,:)];
            end
        else
            ffts_loaded = ffts_loaded(1:end+d,:);
        end
    end
end


function [train_x,train_y] = load_trainfiles_local(trainfilelist_local)
base_dir = 'C:\stuff\masterthesis\';
fftfoldername = strcat(base_dir,'data\fft');%'/media/nikste/moarspace/masterthesis/data/fft';
gtfoldername = strcat(base_dir,'gt');%'/media/nikste/moarspace/masterthesis/gt';
multi_res = 0;
train_y = [];
gts_train_parfor = {}
train_x = [];
ffts_train_parfor = {}
    parfor ind = 1:length(trainfilelist_local)
        
        
        ffts_loaded = importdata(strcat(fftfoldername,'\',trainfilelist_local{ind},'.dataF'));


        %compute multi resolution ffts
        if (multi_res == 1)
            ffts_loaded = createMultiResolutionFFT(ffts_loaded);
        else
            ffts_loaded = normr(sqrt(ffts_loaded));
        end
        %ffts_train = [ffts_train;ffts_loaded];
        %get ground truth ?
        
        gts_loaded = importdata(strcat(gtfoldername,'\',trainfilelist_local{ind},'.dataC'));
        gts_train_parfor{ind}  = gts_loaded(:,1);
    
        %%gts_train = [gts_train;gts_full{ind}];
        
        %subsample

        %this can be done smarter.
    %     end_first_half = floor(size(ffts_loaded,1)/2);
    %     start_second_half = floor( size(ffts_loaded,1)/2+1);
    %     ffts_train = [ffts_train;ffts_loaded(1:2:end_first_half,:);ffts_loaded(start_second_half:4:end,:)];
        
    
        d = size(gts_train_parfor{ind},1) - size(ffts_loaded,1);
        if(d ~= 0)
            if(d > 0) % add frame
                for i=1:d
                    ffts_loaded = [ffts_loaded;ffts_loaded(end,:)];
                end
            else
                ffts_loaded = ffts_loaded(1:end+d,:);
            end
        end
            %ffts_train = [ffts_train;ffts_loaded(1:shrinkfactor:end,:)];
        %    indices = [indices ind];
        ffts_train_parfor{ind} = ffts_loaded;
        %ffts_train = [ffts_train;ffts_loaded];
        %gts_train = [gts_train;gts_full{ind}(1:shrinkfactor:end,1)];
        
        %d = size(gts_full{ind}(1:shrinkfactor:end,:),1) - size(ffts_loaded(1:shrinkfactor:end,:),1);
        %disp(strcat('gts_full:',num2str(size(gts_full{ind}(1:shrinkfactor:end,:),1)),':',num2str(size(ffts_loaded(1:shrinkfactor:end,:),1))));
        %assert(d == 0);
        
    end
    % merge results
    for ind = 1:length(trainfilelist_local)
        train_x = [train_x;ffts_train_parfor{ind}];
        train_y = [train_y;gts_train_parfor{ind}];
    end
    clearvars ffts_train_parfor
    clearvars gts_train_parfor
    %displ(size(train_x,1) - size(train_y,1))
    assert(size(train_x,1) == size(train_y,1))
end



