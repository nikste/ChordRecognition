function [nn_left,nn_middle,nn_right,nn_top]= trainTest(train_x,train_y,nn_left,nn_middle,nn_right,nn_top)
%TRAINTEST Summary of this function goes here
%   Detailed explanation goes here


train_y = convertTo1K(train_y);
%truncate data
val_x = train_x(1:10:end,:);
val_y = train_y(1:10:end,:);

num_batch = 100;
train_x(1:10:end,:) = [];
train_y(1:10:end,:) = [];
overflow = mod(size(train_x,1),num_batch);
val_x = [val_x; train_x((end-overflow):end,:)];
val_y = [val_y; train_y((end-overflow):end,:)];
train_x = train_x(1:(end - overflow),:);
train_y = train_y(1:(end - overflow),:);


left = runNN(nn_left,train_x(1:end,1:1500));
middle = runNN(nn_middle,train_x(1:end,1501:3000));
right = runNN(nn_right,train_x(1:end,3001:4500));

% Train the FFNN for PCP
opts.numepochs =  50;
opts.batchsize = 100;%6203;%2189; %308;%100;
opts.plot = 1;




    
%% copy paste
assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L_top = zeros(numepochs*numbatches,1);
L_left = zeros(numepochs*numbatches,1);
L_middle = zeros(numepochs*numbatches,1);
L_right = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        [nn_left,nn_middle,nn_right,nn_top] = nnff_custom(nn_left,nn_middle,nn_right,nn_top,batch_x,batch_y);
        [nn_left,nn_middle,nn_right,nn_top] = nnbp_custom(nn_left,nn_middle,nn_right,nn_top);
        [nn_left,nn_middle,nn_right,nn_top] = nnapplygrads_custom(nn_left,nn_middle,nn_right,nn_top);
        
        L_top(n) = nn_top.L;
        L_left(n) = nn_left.L;
        L_right(n) = nn_right.L;
        L_middle(n) = nn_middle.L;
        n = n + 1;
    end
    
    t = toc;

    if opts.validation == 1
        % propagate to top level train_x, val_x
        train_x_top = [runNN(nn_left,train_x(:,1:1500)) runNN(nn_left,train_x(:,1501:3000)) runNN(nn_left,train_x(:,3001:4500))];
        val_x_top = [runNN(nn_left,val_x(:,1:1500)) runNN(nn_left,val_x(:,1501:3000)) runNN(nn_left,val_x(:,3001:4500))];
        
        loss = nneval(nn_top, loss, train_x_top, train_y, val_x_top, val_y);
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        % propagate to top level train_x
        train_x_top = [runNN(nn_left,train_x(:,1:1500)) runNN(nn_left,train_x(:,1501:3000)) runNN(nn_left,train_x(:,3001:4500))];
        
        
        loss = nneval(nn_top, loss, train_x_top, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn_top, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L_top((n-numbatches):(n-1)))) str_perf]);
    nn_top.learningRate = nn_top.learningRate * nn_top.scaling_learningRate;
        nn_left.learningRate = nn_left.learningRate * nn_left.scaling_learningRate;
            nn_middle.learningRate = nn_middle.learningRate * nn_middle.scaling_learningRate;
                nn_right.learningRate = nn_right.learningRate * nn_right.scaling_learningRate;
end



end


function [nn_left,nn_middle,nn_right,nn_top] = nnff_custom(nn_left,nn_middle,nn_right,nn_top,batch_x,batch_y)
    nn_left = nnff(nn_left,batch_x(:,1:1500),zeros(size(batch_x,1),200));
    nn_middle = nnff(nn_middle,batch_x(:,1501:3000),zeros(size(batch_x,1),200));
    nn_right = nnff(nn_right,batch_x(:,3001:4500),zeros(size(batch_x,1),200));
    x = [nn_left.a{end} nn_middle.a{end} nn_right.a{end}];
    nn_top = nnff(nn_top,x,batch_y);
end


function [nn_left,nn_middle,nn_right,nn_top] = nnbp_custom(nn_left,nn_middle,nn_right,nn_top)

    %nn_top = nnbp(nn_top);
    
    %% top!
    nn_top.dropoutFraction = 0;
    n = nn_top.n;
    sparsityError = 0;
    switch nn_top.output
        case 'sigm'
            d{n} = - nn.e .* (nn_top.a{n} .* (1 - nn_top.a{n}));
        case {'softmax','linear'}
            d{n} = - nn_top.e;
    end
    for i = (n - 1) : -1 : 1%2
        % Derivative of the activation function
        switch nn_top.activation_function 
            case 'sigm'
                d_act = nn_top.a{i} .* (1 - nn_top.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn_top.a{i}.^2);
        end
        
        if(nn_top.nonSparsityPenalty>0)
            pi = repmat(nn_top.p{i}, size(nn_top.a{i}, 1), 1);
            sparsityError = [zeros(size(nn_top.a{i},1),1) nn_top.nonSparsityPenalty * (-nn_top.sparsityTarget ./ pi + (1 - nn_top.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn_top.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn_top.W{i} + sparsityError) .* d_act;
        end
        
        if(nn_top.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn_top.dropOutMask{i}];
        end
        
        if (i == 1) % for last layer
            last = d{i};
        end
    end

    for i = 1 : (n - 1)
        if i+1==n
            nn_top.dW{i} = (d{i + 1}' * nn_top.a{i}) / size(d{i + 1}, 1);
        else
            nn_top.dW{i} = (d{i + 1}(:,2:end)' * nn_top.a{i}) / size(d{i + 1}, 1);      
        end
    end
    % copy error to underlying networks:
    
        %% left!
    n = nn_left.n;
    sparsityError = 0;
    d{n} = last(:,2:201);
%     switch nn.output
%         case 'sigm'
%             d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
%         case {'softmax','linear'}
%             d{n} = - nn.e;
%     end

    % take first 200 propagated errors
    
    
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn_left.activation_function 
            case 'sigm'
                d_act = nn_left.a{i} .* (1 - nn_left.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn_left.a{i}.^2);
        end
        
        if(nn_left.nonSparsityPenalty>0)
            pi = repmat(nn_left.p{i}, size(nn_left.a{i}, 1), 1);
            sparsityError = [zeros(size(nn_left.a{i},1),1) nn_left.nonSparsityPenalty * (-nn_left.sparsityTarget ./ pi + (1 - nn_left.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn_left.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn_left.W{i} + sparsityError) .* d_act;
        end
        
        if(nn_left.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn_left.dropOutMask{i}];
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn_left.dW{i} = (d{i + 1}' * nn_left.a{i}) / size(d{i + 1}, 1);
        else
            nn_left.dW{i} = (d{i + 1}(:,2:end)' * nn_left.a{i}) / size(d{i + 1}, 1);      
        end
    end

    
    %% middle
    n = nn_middle.n;
    sparsityError = 0;
    d{n} = last(:,2:201);
%     switch nn.output
%         case 'sigm'
%             d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
%         case {'softmax','linear'}
%             d{n} = - nn.e;
%     end

    % take first 200 propagated errors
    
    
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn_middle.activation_function 
            case 'sigm'
                d_act = nn_middle.a{i} .* (1 - nn_middle.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn_middle.a{i}.^2);
        end
        
        if(nn_middle.nonSparsityPenalty>0)
            pi = repmat(nn_middle.p{i}, size(nn_middle.a{i}, 1), 1);
            sparsityError = [zeros(size(nn_middle.a{i},1),1) nn_middle.nonSparsityPenalty * (-nn_middle.sparsityTarget ./ pi + (1 - nn_middle.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn_middle.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn_middle.W{i} + sparsityError) .* d_act;
        end
        
        if(nn_middle.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn_middle.dropOutMask{i}];
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn_middle.dW{i} = (d{i + 1}' * nn_middle.a{i}) / size(d{i + 1}, 1);
        else
            nn_middle.dW{i} = (d{i + 1}(:,2:end)' * nn_middle.a{i}) / size(d{i + 1}, 1);      
        end
    end
    
    
        %% right
    n = nn_right.n;
    sparsityError = 0;
    d{n} = last(:,2:201);
%     switch nn.output
%         case 'sigm'
%             d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
%         case {'softmax','linear'}
%             d{n} = - nn.e;
%     end

    % take first 200 propagated errors
    
    
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn_right.activation_function 
            case 'sigm'
                d_act = nn_right.a{i} .* (1 - nn_right.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn_right.a{i}.^2);
        end
        
        if(nn_right.nonSparsityPenalty>0)
            pi = repmat(nn_right.p{i}, size(nn_right.a{i}, 1), 1);
            sparsityError = [zeros(size(nn_right.a{i},1),1) nn_right.nonSparsityPenalty * (-nn_right.sparsityTarget ./ pi + (1 - nn_right.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn_right.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn_right.W{i} + sparsityError) .* d_act;
        end
        
        if(nn_right.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn_right.dropOutMask{i}];
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn_right.dW{i} = (d{i + 1}' * nn_right.a{i}) / size(d{i + 1}, 1);
        else
            nn_right.dW{i} = (d{i + 1}(:,2:end)' * nn_right.a{i}) / size(d{i + 1}, 1);      
        end
    end
end
function [nn_left,nn_middle,nn_right,nn_top] = nnapplygrads_custom(nn_left,nn_middle,nn_right,nn_top)
nn_left = nnapplygrads(nn_left);
nn_middle = nnapplygrads(nn_middle);
nn_right = nnapplygrads(nn_right);
nn_top = nnapplygrads(nn_top);

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