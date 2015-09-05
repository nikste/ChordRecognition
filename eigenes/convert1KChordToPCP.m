function [pcp] = convert1KChordToPCP(K)

% construct matrix for conversion of chord to best pcp

convMat = zeros(25,12);

% non - chord
convMat(1,:) = [1 1 1 1 1 1 1 1 1 1 1 1];

% majors
root = 1-1;
terz = 5-1;
quint = 8-1;
for note_offset = 0:11
    chord = note_offset + 2;
% 
    root_offset = mod(root + note_offset, 12);
    terz_offset = mod(terz + note_offset, 12);
    quint_offset = mod(quint + note_offset, 12);
% 
     convMat(chord,root_offset + 1) = 1;
     convMat(chord,terz_offset + 1) = 1;
     convMat(chord,quint_offset + 1) = 1;
end

% minors
root = 1 - 1;
terz = 4 - 1;
quint = 8 - 1;
for note_offset = 0:11
    chord = note_offset + 14;

    root_offset = mod(root + note_offset, 12);
    terz_offset = mod(terz + note_offset, 12);
    quint_offset = mod(quint + note_offset, 12);

    convMat(chord,root_offset + 1) = 1;
    convMat(chord,terz_offset + 1) = 1;
    convMat(chord,quint_offset + 1) = 1;
end
convMat = bsxfun(@times, convMat, 1./(sum(convMat, 2)));%for l2 norm : normr(convMat);

%find finds all non zero indices in the matrix K
pcp = convMat(find(K),:);
end
