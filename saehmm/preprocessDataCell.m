%% preprocessDataCell(X)
% preprocesses input fft data if cell
% compresses (sqrt), whitens and normalizes to L2 norm
% X : data to be processed (fft)
% data : output preprocessed data

function [data] = preprocessDataCell(X_cell)
disp('preprocessing data as cell')

%construct big matrix and remember boundaries
X = [];

borders = 1;

currentind = 0;
for song = 1:length(X_cell)
    song_data = X_cell{song};
    for frame = 1:size(song_data,1)
        currentind = currentind + 1;
    end
    borders = [borders currentind +1];
    X = [X;song_data];
end

% compress
X = sqrt(X);

%whiten
C= cov(X);
M= mean(X);
[V,D]= eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
W = bsxfun(@minus, X, M) * P;

% normalize
data_big =  normr(X); %TODO: this is wrong!!

data = {}
% reconstruct
for song = 2:length(X_cell)+1
    data{song - 1} = data_big(borders(song-1):borders(song),:);
end

end