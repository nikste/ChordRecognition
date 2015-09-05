function [sae,nn,im1,im2,im3,err] = autoEncoderTest(train_x,train_y,test_x,test_y)

nn = [];

im3 = [];
im2 = [];
im1 = [];
err = -1;
err_list = [];
abs_list = [];

sae = {};
train_x = double(train_x);%/255;
test_x  = double(test_x);%/255;
train_y = double(train_y);
test_y  = double(test_y);

%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
for  firstlayer = 0:200:201
    for secondlayer = 0:-500:0
        for thirdlayer = 0:-500:0
            
            if(firstlayer > 0 && secondlayer == 0 && thirdlayer == 0 || firstlayer > 0 && secondlayer > 0 && thirdlayer == 0 || firstlayer > 0 && secondlayer > 0 && thirdlayer > 0)
                %subplot(1,1,1);
                [error,abs] = constructAE(train_x,train_y,test_x,test_y,firstlayer,secondlayer,thirdlayer);
                err_list = [err_list error];
                abs_list = [abs_list abs];
                'plotting'
%                 subplot(1,1,1);
                plot([1:length(err_list)],err_list);
%                 subplot(1,2,2);
%                 plot([1:length(abs_list)],abs_list);
                drawnow;
            end
        end

    end    %assert(er < 0.16, 'Too big error');
end



%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% dbn.sizes = [100];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

% %%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
% rand('state',0)
% %train dbn
% dbn.sizes = [200 100 50];
% opts.numepochs =   1;
% opts.batchsize = 100;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% 
% figure;
% subplot(1,3,1);
% im1 = visualize(dbn.rbm{1}.W');
% subimage(abs(im1));
% 
% subplot(1,3,2);
% im2 = visualize(dbn.rbm{2}.W');
% subimage(im2);
% 
% subplot(1,3,3);
% im3 = visualize(dbn.rbm{3}.W');
% subimage(im3);
% drawnow;
% 
% 
% 
% %unfold dbn to nn
% nn = dbnunfoldtonn(dbn, 25);%10);
% 'unfolded'
% nn.activation_function = 'sigm';
% 
% %train nn
% opts.numepochs =    10;
% opts.batchsize =    100;
% opts.plot =         1;
% nn.output =         'softmax';
% nn = nntrain(nn, train_x, train_y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
% 'trained'
end

%constructs stacked auto encoder with maximum 3 layers and trains it
function [error,abs] = constructAE(train_x,train_y,test_x,test_y,firstlayer,secondlayer,thirdlayer)
hh_1 = firstlayer
hh_2 = secondlayer
hh_3 = thirdlayer
error = [];
abs = [];

no_epochs_deep = 5;
no_epochs_back = 5;
batchsize = 100;


if( firstlayer >= 1 && secondlayer >= 1 && thirdlayer >= 1 )
    rand('state',0)
    sae = saesetup([1500 firstlayer secondlayer thirdlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.5;

    sae.ae{2}.learningRate = 1;
    sae.ae{2}.inputZeroMaskedFraction = 0.5;
    sae.ae{2}.activation_function = 'sigm';
    
    sae.ae{3}.learningRate = 1;
    sae.ae{3}.inputZeroMaskedFraction = 0.5;
    sae.ae{3}.activation_function = 'sigm';

    %for j=1:30
    %j=1;
    opts.numepochs =  no_epochs_deep;
    opts.batchsize = batchsize;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);

%     im1=visualize(sae.ae{1}.W{1}(:,2:end)');
%     imwrite(im1,strcat('/home/owner/test_imgs/',num2str(firstlayer),'_',num2str(secondlayer),'_',num2str(thirdlayer),'_1.png'));%drawnow;
%     %visualize(sae.ae{1}.W{1}(:,2:end)');
%     %drawnow expose;
%     subplot(1,3,1);
%     subimage(abs(im1));
% 
%     im2=visualize(sae.ae{2}.W{1}(:,2:end)');
%     imwrite(im2,strcat('/home/owner/test_imgs/',num2str(firstlayer),'_',num2str(secondlayer),'_',num2str(thirdlayer),'_2.png'));
%     %visualize(sae.ae{2}.W{1}(:,2:end)');
%     %drawnow expose;
%     subplot(1,3,2);
%     subimage(im2);
%     
%     im3=visualize(sae.ae{3}.W{1}(:,2:end)');
%     imwrite(im3,strcat('/home/owner/test_imgs/',num2str(firstlayer),'_',num2str(secondlayer),'_',num2str(thirdlayer),'_3.png'));
%     %visualize(sae.ae{3}.W{1}(:,2:end)');
%     %drawnow;
%     subplot(1,3,3);
%     subimage(im3);
% 
%     drawnow;



    %end
    % Use the SDAE to initialize a FFNN
    nn = nnsetup([1500 firstlayer secondlayer thirdlayer 25]);
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    nn.W{3} = sae.ae{3}.W{1};

    % Train the FFNN
    opts.numepochs =   no_epochs_back;
    opts.batchsize = batchsize;%6203;%2189; %308;%100;
    opts.plot = 0;
    nn = nntrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);
    er
    error = er
    %abs = bad;
    % plot error in list, 

end
if( firstlayer >= 1 && secondlayer >= 1 && thirdlayer == 0 )
    rand('state',0)
    sae = saesetup([1500 firstlayer secondlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.5;

    sae.ae{2}.learningRate = 1;
    sae.ae{2}.inputZeroMaskedFraction = 0.5;
    sae.ae{2}.activation_function = 'sigm';
    % 
    % sae.ae{3}.learningRate = 1;
    % sae.ae{3}.inputZeroMaskedFraction = 0.5;
    % sae.ae{3}.activation_function = 'sigm';

    %for j=1:30
    %j=1;
    opts.numepochs = no_epochs_deep;
    opts.batchsize = batchsize;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);

%     im1=visualize(sae.ae{1}.W{1}(:,2:end)');
%     imwrite(im1,strcat('/home/owner/test_imgs/',num2str(firstlayer),'_',num2str(secondlayer),'_',num2str(thirdlayer),'_1.png'));%drawnow;
%     %visualize(sae.ae{1}.W{1}(:,2:end)');
%     %drawnow expose;
%     subplot(1,2,1);
%     subimage(abs(im1));
% 
%     im2=visualize(sae.ae{2}.W{1}(:,2:end)');
%     imwrite(im2,strcat('/home/owner/test_imgs/',num2str(firstlayer),'_',num2str(secondlayer),'_',num2str(thirdlayer),'_2.png'));
%     %visualize(sae.ae{2}.W{1}(:,2:end)');
%     %drawnow expose;
%     subplot(1,2,2);
%     subimage(im2);
%     % 
%     % im3=visualize(sae.ae{3}.W{1}(:,2:end)');
%     % imwrite(im3,strcat('/home/owner/test_imgs/',num2str(j),'_3.png'));
%     % %visualize(sae.ae{3}.W{1}(:,2:end)');
%     % %drawnow;
%     % subplot(1,3,3);
%     % subimage(im3);
% 
%     drawnow;



    %end
    % Use the SDAE to initialize a FFNN
    nn = nnsetup([1500 firstlayer secondlayer 25]);
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    nn.W{2} = sae.ae{2}.W{1};
    % nn.W{3} = sae.ae{3}.W{1};

    % Train the FFNN
    opts.numepochs =  no_epochs_back;
    opts.batchsize = batchsize;%6203;%2189; %308;%100;
    opts.plot = 0;
    nn = nntrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);
    er
    error = er
    %abs = bad

end
if( firstlayer >= 1 && secondlayer == 0 && thirdlayer == 0 )
    rand('state',0)
    sae = saesetup([1500 firstlayer]);
    sae.ae{1}.activation_function       = 'sigm';%'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = 0.5;

    % sae.ae{2}.learningRate = 1;
    % sae.ae{2}.inputZeroMaskedFraction = 0.5;
    % sae.ae{2}.activation_function = 'sigm';
    % 
    % sae.ae{3}.learningRate = 1;
    % sae.ae{3}.inputZeroMaskedFraction = 0.5;
    % sae.ae{3}.activation_function = 'sigm';

    %for j=1:30
    %j=1;
    opts.numepochs =  no_epochs_deep;
    opts.batchsize = batchsize;%74436;%2189;%100
    sae = saetrain(sae, train_x, opts);

    im1=visualize(sae.ae{1}.W{1}(:,2:end)');
    imwrite(im1,strcat('/home/owner/test_imgs/',num2str(firstlayer),'_',num2str(secondlayer),'_',num2str(thirdlayer),'_1.png'));%drawnow;
    visualize(sae.ae{1}.W{1}(:,2:end)');
    drawnow;
    %drawnow expose;
%     subplot(1,2,1);
%     subimage(abs(im1));
% 
%     % im2=visualize(sae.ae{2}.W{1}(:,2:end)');
%     % imwrite(im2,strcat('/home/owner/test_imgs/',num2str(j),'_2.png'));
%     % %visualize(sae.ae{2}.W{1}(:,2:end)');
%     % %drawnow expose;
%     % subplot(1,3,2);
%     % subimage(im2);
%     % 
%     % im3=visualize(sae.ae{3}.W{1}(:,2:end)');
%     % imwrite(im3,strcat('/home/owner/test_imgs/',num2str(j),'_3.png'));
%     % %visualize(sae.ae{3}.W{1}(:,2:end)');
%     % %drawnow;
%     % subplot(1,3,3);
%     % subimage(im3);
% 
%     drawnow;



    %end
    % Use the SDAE to initialize a FFNN
    nn = nnsetup([1500 firstlayer 25]);
    nn.activation_function              = 'sigm';%'sigm';
    nn.output                           = 'softmax';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};
    % nn.W{2} = sae.ae{2}.W{1};
    % nn.W{3} = sae.ae{3}.W{1};

    % Train the FFNN
    opts.numepochs =   no_epochs_back;
    opts.batchsize = batchsize;%6203;%2189; %308;%100;
    opts.plot = 1;
    nn = nntrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);
    er
    error = er
    %abs = bad
    
end

end