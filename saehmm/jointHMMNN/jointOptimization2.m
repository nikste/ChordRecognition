function [ mus,sigmas,mixmats,nn ] = jointOptimization2( ffts_train_songs,gts_train_songs, inits,transmat,mus,sigmas,mixmats,nn )
%JOINTOPTIMIZATION2 Summary of this function goes here
%   Detailed explanation goes here


%training iterations
for i=1:10
   
    for song = 1:size(ffts_train_songs,2)
        ffts_train = ffts_train_songs{song};
        gts_train = gts_train_songs{song};
        pcps_train = runNN(nn,ffts_train);  
        disp(strcat('computing for song:',num2str(song), ' of ', num2str(size(ffts_train_songs,2))));
        
        tic
        for t = 1:size(pcps_train,1)
            
            
            state = gts_train(t) + 1;
            % compute gradient
            
            % derivate mixture:
            bt_s = zeros(12,1);
            % derivate emission
            BT = 0;
            x = pcps_train(t,:)';
            %[B,B2] = mixgauss_prob(x, mus(:,state,:), sigmas(:,:,state,:), mixmats(state,:));
            
            no_comps = size(mus(1,1,:),3);
            % for all components:
            for k = 1:no_comps
                x = pcps_train(t,:)';
                m = mus(:,state,k);
                co = sigmas(:,:,state,k);
                inv_co = inv(co);
                w = mixmats(state,k);
                
                % compute emission
                BT = BT + w / sqrt((2*pi)^12 * det(co)) * exp(-1/2 *(x' - m') * inv_co * ( x' - m')');
                
%                 BT = 1;
%                 bt_s = bt_s + w * (m - x);
                % compute derivative

                fac = zeros(1,12);
                for j = 1:12
                    for l=1:12
                        fac(j) = fac(j) + inv_co(j,l) * (m(l) - x(l));
                    end
                    bt_s(j) = bt_s(j) + w / sqrt((2*pi)^12 * det(co))*fac(j) * exp(-1/2 * (x' - m') * inv_co * (x' - m')');
                end
                
            end
            
            % update neural network weights:
            nn.e = ( 1 / BT * bt_s)';
            nn = nnff(nn, ffts_train(t,:), zeros(1,12));% pcps_train(t:t+batchsize-1,:));
            nn.e = ( 1 / BT * bt_s)';
            nn = nnbp(nn);
            nn = nnapplygrads(nn); 
        
            if(mod(t-1,10)==0)   

            end
%             state     
            if(size(pcps_train,1) == t)            
                a=subplot(7,1,1);
                title(a,'derivatives')
                imagesc(bt_s')

                b=subplot(7,1,2);
                title(b,'BT')
                imagesc(BT)

                c =subplot(7,1,3);
                title(c,'error')
                imagesc((1/BT * bt_s)')

                d=subplot(7,1,4);
                title(d,'nn output')
                imagesc(runNN(nn,ffts_train(t,:)))
                
                e=subplot(7,1,5);
                title(e,'idealistic output')
                imagesc(convert1KChordToPCP(convertTo1K(state-1)))
                f=subplot(7,1,6);
                title(f,'weights bottom')
                visualize(nn.W{1})
                g=subplot(7,1,7);
                title(g,'weights top')
                visualize(nn.W{2});

                drawnow
            end            
            
        end % frame in song
    end % song
    toc
    
    %% compute all frames of all songs with nn
    %nn_b.W{2} = nn.W{1};
    %nn = nn_b;
    
    ffts_train = [];
    gts_train = [];
    for song=1:size(ffts_train_songs,2)
        ffts_train = [ffts_train ;ffts_train_songs{song}];
        gts_train = [gts_train; gts_train_songs{song}];
        
    end
    
    
    
    disp('computing neural network output')
    pcps_train = runNN(nn,ffts_train);

    figure
    imagesc(pcps_train);
    drawnow
    
    %sort chords
    pcps_train_chords = sortChords( pcps_train, gts_train );
    
    %% train hmm on nn output
    disp('training gmms');
    num_comps = length(mus(1,1,:));
    for c = 1:length(pcps_train_chords)
       state = c;
       disp(strcat('training gmm for chord:',num2str(c)));
       current_train = pcps_train_chords{c};
       gammas = [];
       sum_gammas = zeros(1,num_comps);
       sum_gammas_total = 0;
       % go trough alll training frames:
       for t=1:size(current_train,1)
          % compute gamma term for current chord/state
          % i.e. weigth of the gaussian compared to the others.
          gammas_current = [];
          gammas_sum = 0;
          for k=1:num_comps
                [R,err] = cholcov(sigmas(:,:,state,k),0);
                while(err ~= 0)
                    disp(strcat('CAUTION! covariance not correct:',mat2str(sigmas(:,:,state,k))));
                    sigmas(:,:,state,k) = sigmas(:,:,state,k) + rand([size(sigmas,1) size(sigmas,2)]) .* eye(12) * 0.0001;
                    disp(strcat('added noise:',mat2str(sigmas(:,:,state,k))));
                    [R,err] = cholcov(sigmas(:,:,state,k),0);
                end
              gamma =  mixmats(state,k) * mvnpdf(current_train(t,:),mus(:,state,k)',sigmas(:,:,state,k));
              gammas_sum = gammas_sum + gamma;
              gammas_current = [gammas_current gamma];
          end
          gammas = [gammas;gammas_current ./ repmat(gammas_sum,1,num_comps)]; % for this time this component
          sum_gammas = sum_gammas + gammas_current / gammas_sum; % over all time this components gamma
          sum_gammas_total = sum_gammas_total + sum(sum_gammas); % over all time all components gamma
       end
       
       sum_w = zeros(num_comps,1);
       sum_mu = zeros(length(mus(:,1,1)),num_comps);
       sum_cov = zeros([size(sigmas(:,:,1,1)),num_comps]);
       % compute parameters for this state:
       for t=1:size(current_train,1)
           for k = 1:num_comps
               sum_w(k) = sum_w(k) + gammas(t,k); % sum over all time this component
               
               sum_mu(:,k) = sum_mu(:,k) + gammas(t,k) * current_train(t,:)'; % sum over all time this component
                
               d =  (current_train(t,:)' - mus(:,state,k) ) ;
               sum_cov(:,:,k) = sum_cov(:,:,k) + gammas(t,k) * d * d';  % sum over all time this component
           
           end
       end
       
       for k=1:num_comps
          sum_w(k) = sum_w(k) /  size(current_train,1);%sum_gammas_total;
          
          sum_mu(:,k) = sum_mu(:,k) / sum_gammas(k);
          sum_cov(:,:,k) = sum_cov(:,:,k) / sum_gammas(k);%+ eye(size(sigmas,1)) * 0.01;
       end

       mixmats(c,:) = sum_w';%weights;       
       mus(:,c,:) =sum_mu;% mu;
       
       sigmas(:,:,c,:) = sum_cov ;%sigma;
       
       %toc
    end
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