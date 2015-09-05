%% preprocessData(X)
% preprocesses input fft data
% compresses (sqrt), whitens and normalizes to L2 norm
% X : data to be processed (fft)
% data : output preprocessed data

function [data,M,P] = preprocessData(X)
disp('preprocessing data')
% compress
X = sqrt(X);

%whiten
C= cov(X);
M= mean(X);
[V,D]= eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.00001))) * V';
W = bsxfun(@minus, X, M) * P;


% make positive ?!
%W = W + ones(size(W)) * abs(min(min(W)));
% normalize
data =  normr(W);%X);

end