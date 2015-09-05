function [nn, L]  = nntrain_earlystop2(nn, opts, id)%(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

nn.inputZeroMaskedFraction = 0.7;
nn.dropoutFraction = 0.7;
nn.sparsityTarget = 0.05;
nn.nonSparsityPenalty = 0.1;
nn.dropoutFraction = 0.5;
partitions = 8;


loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];


opts.validation = 1;


fhandle = [];
fhandle = figure();




batchsize = opts.batchsize;
numepochs = opts.numepochs;


filelist = loadFilenames(id);






min_loss = 100000;
min_net = nn;
counter = 0;
i_inc = 0;
for i = 1 : numepochs
    
    
    for p = 1:partitions
    
        filelist_local = filelist(p : partitions : end); 
        [train_x,train_y] = loadFiles(filelist_local);
        
        assert(size(train_x,1) == size(train_y,1),'train_x and train_y do not have the same size!!');
        
        %%TODO: preprocess!!
        train_y = convertTo1K(train_y);
        
        last_point = size(train_x,1) - mod(size(train_x,1),batchsize);
        val_x = train_x(last_point : end,:);
        val_y = train_y(last_point : end,:);
        train_x = train_x(1:last_point,:);
        train_y = train_y(1:last_point,:);
        
        
        
        m = size(train_x, 1);
        numbatches = m / batchsize;

        assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

        L = zeros(numepochs*numbatches,1);
        n = 1;
    
        tic;

        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);

            %Add noise to input (for use in denoising autoencoder)
            if(nn.inputZeroMaskedFraction ~= 0)
                %set only to zero if smaller than inputZeroMaskedFraction.
                batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
                % isotropic gaussian noise
                %mak = (rand(size(batch_x))<nn.inputZeroMaskedFraction);
                %noise = ( randn(size(batch_x)) * chol(2) ) .* mak;
                %batch_x_old = batch_x;
                %batch_x = batch_x +  noise;
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
            nnupdatefigures_incremental(nn,fhandle,loss,opts,i_inc)
%             save_epochs = opts.numepochs;
%             opts.numepochs = i;
%             %nnupdatefigures(nn, fhandle, loss, opts, i);
%             opts.numepochs = save_epochs;
        end

        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
        nn.learningRate = nn.learningRate * nn.scaling_learningRate;


        % save value for early stopping:
        if(loss.val.e_frac(i) < min_loss)
            disp('counter reset');
            min_loss = loss.val.e_frac(i);
            min_net = nn;
            counter = 0;
        else
            counter = counter + 1;
            disp(strcat('counter:',num2str(counter)));
        end

        %stop if not increasing performance
        if(counter >= 40 || i > 500)
            disp('breaking not increasing!!');
            nn = min_net;
            break;
        end
    end
end
end


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
fftfoldername = 'C:\stuff\masterthesis\data\fft';
train_x = [];
for ind = 1 : size(filelist_local,2)
    filename = strcat(fftfoldername,'\',filelist_local{ind},'.dataF');
    %disp(['loading file:' filename ])
    train_x_ = load(filename,'-mat');
    train_x_ = train_x_.ffts_train;
    train_x = [train_x;train_x_];
end

gtfoldername = 'C:\stuff\masterthesis\gt';
train_y = [];
for ind = 1 : size(filelist_local,2)
    filename = strcat(gtfoldername,'\',filelist_local{ind},'.dataC');
    %disp(['loading file:' filename ])
    train_y_ = importdata(strcat(gtfoldername,'\',filelist_local{ind},'.dataC'));
    train_y_ = train_y_(:,1);
    train_y = [train_y;train_y_];
end

end