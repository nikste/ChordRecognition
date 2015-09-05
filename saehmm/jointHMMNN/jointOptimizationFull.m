function [ mus,sigmas,mixmats,nn ] = jointOptimizationFull( ffts_train_songs,gts_train_songs, inits,transmat,mus,sigmas,mixmats,nn )
%JOINTOPTIMIZATION uses bengio training
%   ffts_trian is cell chord, frame has to be sorted for chords
    %% compute all frames of all songs with nn
    
    %disp(strcat('computing neural network output for song:',num2str(song))
    %pcps_train = runNN(nn,ffts_train);
%     pcps_train = sortChords( pcps_train , gts_train );
%     pcps_train = {};
%     for c = 1:length(ffts_train)
%         song_data = ffts_train{c};
%         %compute neural network output
%         [res] = runNN(nn, song_data);
%         %plugin back
%         pcps_train{c} = res;
%     end
% 
%     %% train hmm on nn output
%     disp('training gmms');
%     for c = 1:length(pcps_train)
%        %get all data for current state
%        current_train = pcps_train{c};  
% 
%        % do train shit
%        disp(strcat('training gmm for chord:',num2str(c)));
%        tic
%        %[mu, Sigma, weights] = mixgauss_init(number_mixtures, current_train, 'full', 'kmeans')
%        [mu, sigma, weights] = mixgauss_em(current_train', number_components, 'verbose', 0,'method', 'kmeans');
% 
%        % save params for current mixture shit
%        mus(:,c,:) = mu;
%        sigmas(:,:,c,:) = sigma;
%        mixmats(c,:) = weights;
%        toc
%     end

figure;
max_iter = 1;
for i=1:max_iter
    disp(strcat('training iteration:',num2str(i), ' of ',num2str(max_iter)));
    for song = 1:size(ffts_train_songs,2)    
        disp(strcat('computing emission probabilities of hmm states for song ',num2str(song),'/',num2str(size(ffts_train_songs,2))));

       

        %r = randperm(size(ffts_train,1));
        %ffts_train = ffts_train(r,:);
        %gts_train = gts_train(r,:);
        %pcps_train = pcps_train(r,:);
        %tic
        % compute emission probability for all frames:
        %[B,B2] = mixgauss_prob(pcps_train',mus,sigmas,mixmats);
        %toc
        % B2(i,k,t) = i=state,k=component,t=timestep
        % 

        % backpropagate all the frames!

        disp('computing gradients')
        tic
        batchsize = 1;

        ffts_train = ffts_train_songs{song};
        gts_train = gts_train_songs{song};
        pcps_train = runNN(nn,ffts_train);
        for t = 1:size(pcps_train,1)-1000

            % compute delta b_t Y_t as in bengio
            % (sum_l d_klj(mu_kl - Ylt))
            % d_klj is the element (l,j) of the inverse of the covariance matrix
            % (Sigma^-1) for the k^th gaussian distribution and mu_kl is the lth
            % element of the k^th gaussian mean vector mu_k

            % thats essentially the derivative of gaussian mixture model


            %pcps_train = runNN(nn,ffts_train);
            %batchmode!
            bt_s_big = zeros(12,25);
            BT_big = zeros(25,1);
            errors = zeros(batchsize,12);
            errors_big = zeros(12,25);
            %expectation of mixture model, weighted sum of means
            targets = zeros(batchsize,12);



%            current_batch = pcps_train(t:t+batchsize-1,:);%runNN(nn,ffts_train(t:t+batchsize-1,:));

                % for all components:
                %state = gts_train(b)+1;
                no_comps = length(mus(1,1,:));
                BT = 0;


                %pcps_train = runNN(nn,ffts_train);
                [B, B2] = mixgauss_prob(pcps_train', mus, sigmas, mixmats);

                [alpha, beta, gamma, loglik, xi_summed, gamma2] = fwdback(inits,transmat, B);
                % For HMMs with MOG outputs: if you want to compute gamma2, you must specify
                % 'obslik2' - obslik(i,j,t) = Pr(Y(t)| Q(t)=i,M(t)=j)  []
                % 'mixmat' - mixmat(i,j) = Pr(M(t) = j | Q(t)=i)  []




        %         BT2 = 0;
        %         bt_s2 = zeros(12,1);
                %disp(num2str(state));
                
                %for all states
                for state=1:25
                    bt_s = zeros(12,1);
                    %act = 0;
                    for k = 1:no_comps


                        %compute activation for this component
                        x = pcps_train(t,:);%current_batch(t,:);%pcps_train(b,:);
                        m = mus(:,state,k)';
                        co = sigmas(:,:,state,k);
                        w = mixmats(state,k);

                        %"target"
                        targets(t,:) = targets(t,:) + w * m;

                        inv_co = inv(co);
                        %act = act + w/(sqrt((2*pi))^12 * sqrt(det(co))) * exp(-1/2 *(x - m) * inv_co * ( x- m)');
                        for j = 1:size(bt_s,1)
                            fac = 0;
                            for l=1:size(co,1)
                                fac = fac + inv_co(l,j) * (m(l) - x(l));
                            end
                            bt_s(j) = bt_s(j) +  w/(sqrt((2*pi)^12 * det(co))) *fac* exp(-1/2 *(x - m) * inv_co * ( x- m)');    
                        end
            %             act = w * mvnpdf(x,m,co);
            %             %activation for all components
            %             BT2 = BT2 + act;
            %             
            %             inv_cov = inv(sigmas(:,:,state,k));
            %             d = (m - x);
            %             
            %             bt_s2 = bt_s2 + w * mvnpdf(x,m,co) * inv_cov * d'; 

                    end
                    BT = gamma(state,t) / B(state,t);%act;
                    bt_s_big(:,state) = bt_s;
                    BT_big(state) = BT;
                    errors_big(:,state) = BT*bt_s;

                    
                    %errors(b-t+1,:) = errors(b-t+1) + BT * bt_s;
                    %dirty hack:
                    %errors(b-t+1,:) = errors(b-t+1,:)/sum(abs(errors(b-t+1,:)));
                end
                    if(sum(abs(errors)) > 10)
                        disp(errors);
                    end
         
            if(mod(t,100) == 0)
                disp('100')
            end
            errors = -sum(errors_big,2);
            if(mod(t,1)==0)            
            a=subplot(7,1,1);
            title(a,'derivatives')
            imagesc(bt_s_big')

            b=subplot(7,1,2);
            title(b,'weights')
            imagesc(BT_big)

            c =subplot(7,1,3);
            title(c,'error')
            imagesc(-sum(errors_big,2)')

            d=subplot(7,1,4);
            title(d,'nn output')
            imagesc(runNN(nn,ffts_train(t:t+batchsize-1,:)))
            e=subplot(7,1,5);
            title(e,'idealistic output')
            imagesc(convert1KChordToPCP(convertTo1K(gts_train(t))))
            f=subplot(7,1,6);
            title(f,'weights bottom')
            visualize(nn.W{1})
            g=subplot(7,1,7);
            title(g,'weights top')
            visualize(nn.W{2});
            
            
%             subplot(10,1,8);
%             plot(mus(:,gts_train(t)+1,1))
%             subplot(10,1,9);
%             plot(mus(:,gts_train(t)+1,2))
%             subplot(10,1,10);
%             plot(mus(:,gts_train(t)+1,3))
            
            
            
            drawnow
            
            
            
            
            %pause(0.25)
            end
    

            %nn.e = -(1/BT * bt_s)';
            nn.e = -(sum(errors_big,2)/sum(sum(abs(errors_big))))'; %-(1/BT_big * bt_s_big')
            nn.e = -sum(errors_big,2)';
            nn.e
            %normr(nn.e)
            nn = nnffjoint(nn, ffts_train(t:t+batchsize-1,:),targets);% pcps_train(t:t+batchsize-1,:));
            nn = nnbp(nn);
            nn = nnapplygrads(nn); 
            
            
            % TODO: batch gradient?
    % %         state = gts_train(t)+1;
    % %         s = zeros(size(pcps_train,2),1);
    % % %         [B, B2] = mixgauss_prob(pcps_train(t,:)', mus, sigmas, mixmats);
    % %         BT = 0;
    % %         for k = 1:length(mus(1,1,:))
    % %             %disp(strcat('t:',num2str(t),' comp:',num2str(k),' state:',num2str(state)));
    % % 
    % %             x = pcps_train(t,:);
    % %             m = mus(:,state,k)';
    % %             co = sigmas(:,:,state,k);
    % %             act = mixmats(state,k) * mvnpdf(x,m,co);
    % %            BT = BT+ act;
    % %            
    % %            inv_cov = inv(sigmas(:,:,state,k));
    % %            s2 = zeros(1,12);
    % %            for l=1:12
    % %               s2 = s2 + inv_cov(l,:) * (mus(l,state,k) - pcps_train(t,l))';
    % %            end
    % %            
    % %            %mus(:j,k) j:state k:mixture
    % %            s3 = inv_cov * (mus(:,state,k) - pcps_train(t,:)');
    % %            s = s + act * s3; %mixmats(state,k)*( s3 * mvnpdf(x,m,co));
    % %         end
    % % %         BT = B(state);
    % % %         s = 0;
    % % %         for k=1:length(mus(1,1,:))
    % % %             inv_cov = inv(sigmas(:,:,state,k));
    % % %             s = s + B2(state,k) * mixmats(state,k) * inv_cov * (mus(:,state,k) - pcps_train(t,:)' ); 
    % % %         end
    % %         nn.e = 1/BT * s';        
    % %         nn = nnffjoint(nn, ffts_train(t,:), pcps_train(t,:));
    % %         % backpropagate this
    % %         % TODO: check if B or sum(B2) ?!
    % %         nn.e = -1/BT * s';
    % %         %disp(strcat('error:',mat2str(nn.e)));
    % % %         hope tis it
    % %         
    % %         nn = nnbp(nn);
    % %         nn = nnapplygrads(nn); 

            if(isnan(nn.e))
                disp(stcat('error is nan!', num2str(nn.L), ' error:',mat2str(nn.e)));
            end
            if(isinf(nn.e))
                disp(strcat('error is inf!',num2str(nn.L), ' error:', mat2str(nn.e)));
            end
            if(isnan(nn.L))
                disp(strcat('loss is nan!', num2str(nn.L), ' error:',mat2str(nn.e)));
            end
            if(isinf(nn.L))
                disp(strcat('loss is inf!',num2str(nn.L), ' error:', mat2str(nn.e)));
            end
            if(mod(t,1000)==0)
                disp(strcat('computing:',num2str(t),' out of', num2str(size(pcps_train,1))));
                disp(strcat('current grad:',mat2str(nn.e)));
            end
        end
        toc
    end
    
    
    
    
% %     
% %     
% %     %compute emission probability for this frame from mixture of gaussians
% %     for c = 1:length(pcps_train)
%         % bt = that
%         [B, B2] = mixgauss_prob(pcps_train{c}', mus, sigmas, mixmats);
% 
%         % compute delta b_t Y_j,t as described in bengio
%         % B2(i,k,t): i=state, k component, t timestep
%         % (sum_l d_klj(mu_kl - Ylt))
%         % d_klj is the element (l,j) of the inverse of the covariance matrix
%         % (Sigma^-1) for the k^th gaussian distribution and mu_kl is the lth
%         % element of the k^th gaussian mean vector mu_k
% 
% 
%         state = c;
%         for t = 1:size(pcps_train{c})
%             sum = zeros(length(mus(:,1,1)));
%             for k = 1:length(B2(state,:,t)) %components
%                 inv_cov = inv(sigmas(:,:,state,k));
%                 fac = zeros(length(B2(state,:,t)));
%                 for l = 1:length(inv_cov)
%                     fac = fac + inv_cov(l,:) * (mus(l,state,k))
%                 end
%                 sum = sum + fac * B2(state,k,t);
%             end
% 
%         end
%     end
   
%     for song = 1:size(ffts_train_songs)
%         ffts_train = ffts_train_songs{song};
%         gts_train = gts_train_songs{song};
        %% compute all frames of all songs with nn
        disp(strcat('computing neural network output for song ',num2str(song)))
        ffts_train = [];
        gts_train = [];
        for song = 1:size(ffts_train_songs,2)
            ffts_train = [ffts_train; ffts_train_songs{song}];
            gts_train = [gts_train; gts_train_songs{song}];
        end
        pcps_train = runNN(nn,ffts_train);

        figure
        subplot(1,2,1)
        imagesc(pcps_train);
        subplot(1,2,2)
        imagesc(convert1KChordToPCP( convertTo1K(gts_train)));
        drawnow

        %sort chords
        pcps_train_chords = sortChords( pcps_train, gts_train_songs );

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
                    if(err ~= 0)
                        disp(strcat('CAUTION! covariance not correct:',mat2str(sigmas(:,:,state,k))));
                        sigmas(:,:,state,k) = sigmas(:,:,state,k) + rand([size(sigmas,1) size(sigmas,2)]) .* eye(12) * 0.000001;
                        disp(strcat('added noise:',mat2str(sigmas(:,:,state,k))));
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
              sum_cov(:,:,k) = sum_cov(:,:,k) / sum_gammas(k)+ eye(size(sigmas,1)) * 0.01;
           end

           %sum_w should sum to 1!!!
    %        s_w = 0;
    %        for k=1:num_comps
    %            s_w = s_w + sum_w(k);
    %        end
    %        if(s_w ~= 1)
    %            disp(strcat('s_w not 1:',num2str(s_w),' : ',mat2str(sum_w)));
    %        end
           %get all data for current state
    %        current_train = pcps_train_chords{c};  
    %        [B, B2] = mixgauss_prob(pcps_train_chords{c}', mus, sigmas, mixmats);
    %        s = sum(B2(c,:,:));
    %        su = [];
    %        for k=1:length(mus(1,1,:))
    %             su = [su s];
    %        end
    %        Bs =  B2(c,:,:) ./ su;
    %        Bs = squeeze(Bs);
    %        weights = sum(Bs,2)/sum(sum(Bs));
    %        
    %        mu = (Bs*pcps_train_chords{c}./repmat(sum(Bs,2),1,12))';
    % 
    %        sigma = [];
    %        tic
    %        for k=1:length(mus(1,1,:))
    %            sum_k = zeros(12,12);
    %             for t=1:size(Bs,2)
    %                 sum_k = sum_k + Bs(k,t)*((current_train(t,:)' - mus(:,c,k)) * (current_train(t,:)' - mus(:,c,k))');
    %             end
    %             sigma = cat(3,sigma,sum_k/sum(Bs(k,:)));
    %        end
    %        toc
           %sigma = sum(Bs .* (current_train - mu) * (current_train - mu)')./sum(Bs);
           % do train shit
           %disp(strcat('training gmm for chord:',num2str(c)));
           %tic
           %[mu, Sigma, weights] = mixgauss_init(number_mixtures, current_train, 'full', 'kmeans')
           %[mu, sigma, weights] = mixgauss_em(current_train', number_components, 'verbose', 0,'method', 'kmeans');

           % save params for current mixture shit

           %disp(strcat('diff ws:',mat2str(mixmats(c,:) - sum_w')));
           %disp(strcat('diff mus:',mat2str(squeeze(mus(:,c,:)) - sum_mu)));
           %disp(strcat('diff cov:',squeeze(sigmas(:,:,c,:)) - sum_cov));

    %        if(sum(mixmats(c,:) - sum_w') > 0.1)
    %            disp(strcat('mixmats differ:',mat2str(mixmats(c,:)),'old:',mat2str(sum_w')));
    %        end
    %        if(sum(sum(abs(squeeze(mus(:,c,:))) - abs(sum_mu))) > 0.1)
    %            disp(strcat('mus differ:',mat2str(mus(:,c,:)),'old:',mat2str(sum_mu')));
    %        end
    %        if(sum(sum(sum(squeeze(abs(sigmas(:,:,c,:))) - abs(sum_cov)))) > 0.1)   
    %            disp(num2str(sum(sum(sum(abs(squeeze(sigmas(:,:,c,:))) - abs(sum_cov))))))
    %            disp(strcat('sigmas differ:',mat2str(sigmas(:,:,c,1)),'old:',mat2str(sum_cov(:,:,1))));
    %        end
           mixmats(c,:) = sum_w';%weights;       
           mus(:,c,:) =sum_mu;% mu;

           sigmas(:,:,c,:) = sum_cov ;%sigma;



           %toc
        end
%     end
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

