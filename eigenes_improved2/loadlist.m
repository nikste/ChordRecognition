function [fftFiles,gtFiles] = loadlist(inputlist)

org = importdata(inputlist)

fftFiles = {}
gtFiles = {}
for i=1:size(org,1)
    [path,name,ext] = fileparts(org{i});
    fftFiles{i} = strcat('C:\stuff\ChordRecognition\datasets\scratchdir\fft\',name,'.dataF');
    gtFiles{i} = strcat('C:\stuff\ChordRecognition\datasets\scratchdir\gt\',name,'.dataC');
end
end