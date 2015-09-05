function [COEFF,ffts_train_pca,m] = preprocessDataPCA(X)
    

% normal preprocessing
m = mean(X);
[COEFF,SCORE,latent] = princomp(X);

ffts_train_pca = SCORE(:,1:1200);

% depreciate in pca space all after 
%all(all((bsxfun(@minus,ffts_train,mean(ffts_train))*COEFF - SCORE) < 1e-10))
end