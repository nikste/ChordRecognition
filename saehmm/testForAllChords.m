function [chords_present] =  testForAllChords(gts_train)
%checks if all chord labels are in the training set
chords_present = zeros(217,1);
for ind = 1:size(gts_train,1)
    chords_present(gts_train(ind)+1) = chords_present(gts_train(ind)+1) + 1;
end

for chord = 1:217
    if(chords_present(chord) == 0)
        disp(strcat('CHORD:',num2str(chord),' is not in training data!!'));
    end
end

end