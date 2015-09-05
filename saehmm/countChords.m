function [ chords ] = countChords( chord_sequence )
%countChords counts the number of chords in a chord sequence (matrix)
% outputs a matrix (vector), with the counts of each chord.

chords = zeros(25,1);

for t=1:length(chord_sequence)
    % add one to chord symbol, cause matlab.
    chords(chord_sequence(t)+1,1) = chords(chord_sequence(t)+1,1) + 1;
end



end

