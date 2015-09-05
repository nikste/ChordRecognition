function [C,M,V,D,P,W,W_tst] = whitenData(ffts_quart,ffts_quart_test)

%remember input
l = size(ffts_quart,1)
l_test = size(ffts_quart_test,1)

X = [ffts_quart;ffts_quart_test];

C= cov(X);
M= mean(X);
[V,D]= eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
W = bsxfun(@minus, X, M) * P;
% x = ffts_aux;
% % C = cov(ffts_aux);
% % M = mean(ffts_aux);
% % 
% % [V,D] = eig(C);
% % 
% % P = V * diag(sqrt(1./(diag(D) + 0.1)))*V';
% % 
% % %W_aux = bscfun(@minus,ffts_aux,M) * P;
% % W_aux = bscfun(@minus,ffts_aux,M)*P;
% 
% 
% %http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
% avg = mean(x, 1);     % Compute the mean pixel intensity value separately for each patch. 
% x = x - repmat(avg, size(x, 1), 1);
% 
% sigma = x * x' / size(x, 2);
% 
% [U,S,V] = svd(sigma);
% 
% xRot = U' * x;
% 
% 
% W_aux = xRot;

W = X(1:l,:);
W =  normr(W);
W_tst = X(l+1:l+l_test,:);
W_tst = normr(W_tst);
end