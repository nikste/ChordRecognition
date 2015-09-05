function [ chords_sorted ] = sortChords( ffts_train, gts_train )
%SORTCHORDSMAT sorts the chords according to ground truth and saves them in
% cell chords_sorted
%   ffts_train : training mat(frame,fft_bin)
%   gts_train : training gt mat(frame,gt)

%aggregate to one matrix
ffts_train = [gts_train ffts_train];

A = arrayfun(@(x) ffts_train(ffts_train(:,1) == x, :), unique(ffts_train(:,1)), 'uniformoutput', false);

chords_sorted = cell(25,1);
%remove first column
for c = 1:25
    if(isempty(A{c}))
        disp(strcat('CAUTION: matrix for chord:',num2str(c),' is empty'));
    else
        dat = A{c};
        chords_sorted{c} = dat(:,2:end); 
    end
end

end

