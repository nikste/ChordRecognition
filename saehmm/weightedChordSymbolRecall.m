function [res] = weightedChordSymbolRecall(labfiles,predictions)


en = 0;
den = 0;
for song = 1:length(labfiles)
    res = chordSymbolRecall(labfiles{song},predictions{song});
    len = labfiles{song};
    len = len(end,2);
    en = en + (res * len);
    den = den + len;
end

res = en/den;


end