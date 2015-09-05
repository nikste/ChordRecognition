function [ mus,sigmas,mixmats,nn, train_error ] = jointOptimizationOneLayer( ffts_train_songs,gts_train_songs, inits,transitions,mus,sigmas,mixmats,nn )
%JOINTOPTIMIZATION2 uses Bengios joint optimization scheme for a gaussian
%mixture model with in combination with a neural network
%   input:
% ffts_train_songs: fft frames as cell, per song
% gts_train_songs: ground truths frames as cell, per song
% inits: initial state distribution (actually not needed)
% transitions: transition matrix of the hmm (actually not needed)
% mus: matrix of means (:,state,component)
% sigmas: matrix of covariance matrices (:,:,state,component)
% mixmats: matrix of weights (:,state)
% nn: neural network as computed from library of rasmus berg palm

% output:
% mus, updated, as input
% sigmas, updated, as input
% mixmats, updated, as input
% nn, updated, as input



% code fragement remaining if we want to leave the first layer untouched.
% nn_b = nn;
% nn = nnsetup([200 12]);
% nn.W{1} = nn_b.W{2};
% nn.output = 'softmax';
% nn.activation_function = 'sigm';
% nn.learningRate=0.01;
% nn2 = nnsetup([1500 200]);
% nn2.W{1} = nn_b.W{1};
% nn2.activation_function = 'sigm';


nn.learningRate=0.1;

% variable for plotting 
train_error = [];
% holds error for plot, will be reset every 1000 iterations
train_error_single = [];
%plot counter
frame_ctr = 0;

% batchsize for batchtraining
batchsize = 100;

for i=1:1

    h = figure; % for error plot
    %random subset
    for g = 1:1
        disp(num2str(g));
    r = randi([1, 20])
    % go through every song in training set
    for song = 1:size(ffts_train_songs,2)
        
        % get the fft frames of song
        ffts_train = ffts_train_songs{song};
        % get ground truth
        gts_train = gts_train_songs{song};

        % uncomment if leave first layer fixed
        % pcps_train = runNN(nn,runNN(nn2,ffts_train));   
        
        % compute neural network output (not used at the moment)
        pcps_train = runNN(nn,ffts_train);
        
        %randomize
        p = randperm(size(gts_train,1));
        pcps_train = pcps_train(p,:);
        gts_train = gts_train(p,:);
        
        disp(strcat('song:',num2str(song),' ffts:',num2str(size(ffts_train)),' pcps:',num2str(size(pcps_train))));
        
        %random subset
        s = randi([batchsize,10*batchsize]);
        for t = s:10*batchsize:size(pcps_train,1)-batchsize
        
            % aggregation of error for batch for the neural network
            % operations later
            errors_big = zeros(batchsize,12);
            
            for b = t:t+batchsize-1
                
                % recompute every frame in the batch with updated neural
                % network weights
                pcps_train(b,:) = runNN(nn,ffts_train(b,:));%runNN(nn,runNN(nn2,ffts_train(t,:)));
                
                % get ground truth state for gradient computation, adding
                % one because ground truth originally from 0 to 24,
                % indexing in matlab is 1..25
                state = gts_train(b) + 1;
                
                
                %% compute gradient like bengio

                
                
                % derivate of emission
                BT = 0;
                %x = pcps_train(t,:)';
                %[B,B2] = mixgauss_prob(x, mus(:,state,:), sigmas(:,:,state,:), mixmats(state,:));

                % number of components of gaussian mixture
                no_comps = size(mus(1,1,:),3);
                
                % save derivate mixture:
                bt_s = zeros(12,1);
                % go through all components:
                for k = 1:no_comps
                    
                    % current neural network output
                    x = pcps_train(b,:)';
                    
                    % mean of current component for current state
                    m = mus(:,state,k);
                    
                    % covariance of current component for current state
                    co = sigmas(:,:,state,k);
                    
                    % inverted covariance matrix
                    inv_co = inv(co);
                    
                    % weight for current component for current state
                    w = mixmats(state,k);

                    % compute emission
                    BT = BT + w / sqrt((2*pi)^12 * det(co)) * exp(-1/2 *(x' - m') * inv_co * ( x' - m')');
                    
                    % fac is sum over d_k,lj (m_kl - Ylt), see bengio
                    % derivate of gmm with respect to neural network output
                    % "d_k.lj is the element (l,j) of the inverse of the
                    % covariance matrix (Sigma^{-1}) for the kth gaussian
                    % distribution and m_kl is the lth element of the kth
                    % gaussian mean vector m_k"
                    
                    fac = zeros(1,12);
                    for j = 1:12
                        for l=1:12
                            fac(j) = fac(j) + inv_co(l,j) * (m(l) - x(l));
                        end
                        % multiply this with emission probability of
                        % current state and weight of current component.
                        bt_s(j) = bt_s(j) + w / sqrt((2*pi)^12 * det(co))*fac(j) * exp(-1/2 * (x' - m') * inv_co * (x' - m')');
                    end
                    % save this for batch gradient
                    %errors_big(b-t+1,:) = errors_big(b-t+1,:) + mixmats(state,k) * (m - x)';%errors_big(b-t+1,:) + ( 1 / BT * bt_s)';%errors_big(b-t+1,:) + mixmats(state,k) * (m - x)';%errors_big(b-t+1,:) + ( 1 / BT * bt_s)';
                end
               errors_big(b-t+1,:) = (1/ BT * bt_s)';
            end %batch end
            
            % update neural network weights:
            % this is to ensure network error, most probably can be removed
            
            nn.e = errors_big; %( 1 / BT * bt_s)';
            
            % error for non batch training
            %nn = nnff(nn, runNN(nn2,ffts_train(t,:)), zeros(1,12));% pcps_train(t:t+batchsize-1,:));
            
            % forward propagate through network, we reset the error
            % manually, thus target is here a zero vector
            nn = nnff(nn,ffts_train(t:batchsize+t-1,:),zeros(batchsize,12));
            
            % replace error manually, we do not put in - because the
            % library does this later as in :
            % "    
            % n = nn.n;
            % sparsityError = 0;
            % switch nn.output
            %   case 'sigm'
            %       d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
            %   case {'softmax','linear'}
            %       d{n} = - nn.e;
            %   end
            % " 
            % found in nnbp.m
            nn.e = errors_big;%( 1 / BT * bt_s)';
            
            % use backpropagation of library:
            nn = nnbp(nn);
            % apply gradients the library just computed with custom error:
            nn = nnapplygrads(nn);
            
            
            % plotting function, plots average of absolute error for 1000
            % iterations
            train_error_single = [train_error_single sum(mean(abs(nn.e)))];
            if(mod(frame_ctr,500)==0)
                set(0,'CurrentFigure',h)
                train_error = [train_error mean(train_error_single)];
                %plot(train_error)
                train_error_single = [];
                drawnow;
            end
            frame_ctr = frame_ctr+batchsize;
            
            % more elaborate plotting for debugging:
%             state     
%             if(mod(t,1)==0)            
%                 a=subplot(7,1,1);
%                 title(a,'derivatives')
%                 imagesc(bt_s')
% 
%                 b=subplot(7,1,2);
%                 title(b,'BT')
%                 imagesc(BT)
% 
%                 c =subplot(7,1,3);
%                 title(c,'error')
%                 imagesc((1/BT * bt_s)')
% 
%                 d=subplot(7,1,4);
%                 title(d,'nn output')
%                 imagesc(runNN(nn,runNN(nn2,ffts_train(t,:))))
%                 
%                 e=subplot(7,1,5);
%                 title(e,'idealistic output')
%                 imagesc(convert1KChordToPCP(convertTo1K(state-1)))
%                 f=subplot(7,1,6);
%                 title(f,'weights bottom')
%                 visualize(nn.W{1})
% %                 g=subplot(7,1,7);
% %                 title(g,'weights top')
% %                 visualize(nn.W{2});
% 
%                 drawnow
%             end            
            
        end % frame in song
    end % song
    end
    
    
%     %% compute all frames of all songs with nn
%     
%     % uncomment for leaving first layer untouched
%     %     nn_b.W{2} = nn.W{1};
%     %     nn = nn_b;
% 
%     % re save all training data in a format we can compute with neural
%     % network:
%     ffts_train = cell2mat(ffts_train_songs');
%     gts_train = cell2mat(gts_train_songs');
%     % would decrease amount of training data, is not active at the momoent
%     offset = floor(rand(1))+1;
%     ffts_train = ffts_train(offset:end,:);
%     gts_train = gts_train(offset:end,:);
% 
%     
%     
% 
%     
%     disp('computing neural network output')
%     
%     % recompute neural network output with update neural network
%     pcps_train = runNN(nn,ffts_train);
%     
%     % make plot of outcome to check for useless convergence
%     b = figure;
%     set(0,'CurrentFigure',b)
%     imagesc(pcps_train)
%     drawnow;
%     
%     %sort chords, each cell contains training for chord corresponding to
%     %its index
%     pcps_train_chords = sortChords( pcps_train, gts_train );
%     
%     
%     %% train hmm on nn output
%     disp('training gmms');
%     % number of components to update
%     num_comps = length(mus(1,1,:));
%     
%     % go through all chords
%     for c = 1:length(pcps_train_chords)
%        %current state
%        state = c;
%        
%        disp(strcat('training gmm for chord:',num2str(c)));
%        current_train = pcps_train_chords{c};
%        
%        % update as in rabbiner
%        gammas = [];
%        sum_gammas = zeros(1,num_comps);
%        sum_gammas_total = 0;
%        
%        %subset
%        r = randi([1,10])
%        
%        % go trough alll training frames:
%        for t=r:10:size(current_train,1)
%           % compute gamma term for current chord/state
%           % i.e. weigth of the gaussian compared to the others.
%           gammas_current = [];
%           gammas_sum = 0;
%           
%           % for all components in our hmm
%           for k=1:num_comps
%                 % in case we have a malformed covariance matrix add some
%                 % noise along the diagonal
%                 [R,err] = cholcov(sigmas(:,:,state,k),0);
%                 while(err ~= 0)
%                     disp(strcat('CAUTION! covariance not correct:',mat2str(sigmas(:,:,state,k))));
%                     sigmas(:,:,state,k) = sigmas(:,:,state,k) + rand([size(sigmas,1) size(sigmas,2)]) .* eye(12) * 0.0001;
%                     disp(strcat('added noise:',mat2str(sigmas(:,:,state,k))));
%                     [R,err] = cholcov(sigmas(:,:,state,k),0);
%                 end
%               gamma =  mixmats(state,k) * mvnpdf(current_train(t,:),mus(:,state,k)',sigmas(:,:,state,k));
%               gammas_sum = gammas_sum + gamma;
%               gammas_current = [gammas_current gamma];
%           end
%           gammas = [gammas;gammas_current ./ repmat(gammas_sum,1,num_comps)]; % for this time this component
%           sum_gammas = sum_gammas + gammas_current / gammas_sum; % over all time this components gamma
%           sum_gammas_total = sum_gammas_total + sum(sum_gammas); % over all time all components gamma
%        end
%        
%        sum_w = zeros(num_comps,1);
%        sum_mu = zeros(length(mus(:,1,1)),num_comps);
%        sum_cov = zeros([size(sigmas(:,:,1,1)),num_comps]);
%        % compute parameters for this state:
%        for t=1:size(current_train,1)/10
%            for k = 1:num_comps
%                sum_w(k) = sum_w(k) + gammas(t,k); % sum over all time this component
%                
%                sum_mu(:,k) = sum_mu(:,k) + gammas(t,k) * current_train(t,:)'; % sum over all time this component
%                 
%                d =  (current_train(t,:)' - mus(:,state,k) ) ;
%                sum_cov(:,:,k) = sum_cov(:,:,k) + gammas(t,k) * d * d';  % sum over all time this component
%            
%            end
%        end
%        
%        % weights, means and covariances
%        for k=1:num_comps
%           sum_w(k) = sum_w(k) /  size(current_train,1);%sum_gammas_total;
%           
%           sum_mu(:,k) = sum_mu(:,k) / sum_gammas(k);
%           sum_cov(:,:,k) = sum_cov(:,:,k) / sum_gammas(k)+ eye(size(sigmas,1)) * 0.01;
%        end
% 
%        mixmats(c,:) = sum_w';%weights;       
%        mus(:,c,:) =sum_mu;% mu;
%        
%        sigmas(:,:,c,:) = sum_cov ;%sigma;
%        
%        %toc
%     end
    % re save all training data in a format we can compute with neural
    % network:
    ffts_train = cell2mat(ffts_train_songs');
    gts_train = cell2mat(gts_train_songs');
    % would decrease amount of training data, is not active at the momoent
    offset = floor(rand(1))+1;
    ffts_train = ffts_train(offset:end,:);
    gts_train = gts_train(offset:end,:);

[pcps] = runNN(nn,ffts_train);
% pcps = rand(size(ffts_train,1),12);
% disp(strcat('pcps:',num2str(size(pcps))))
% disp(strcat('gst_train:',num2str(size(gts_train))))
[chords_sorted_pcp] = sortChords( pcps, gts_train );


%make it smaller
for c=1:25
    r = randi([1,10]);
chords_sorted_pcp{c} = chords_sorted_pcp{c}(r:10:end,:);
end

%% gaussian mixture hmm
disp('training hmm on neural network output')
number_components = 2 ;
disp(strcat('number of components:',num2str(number_components)));
[inits,transitions, mus,sigmas,mixmats ] = train_gmms( chords_sorted_pcp ,gts_train_songs, number_components);



end %training iterations
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

