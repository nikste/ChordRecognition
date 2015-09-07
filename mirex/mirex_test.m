function [res] = mirex_test(trainlistfile,scratchdir,resultdir)


% For testing: 
% doChordID.sh "/path/to/testFileList.txt" "/path/to/scratch/dir" "/path/to/results/dir" 
% If there is no training, you can ignore the second argument here. In the results directory, there
% should be one file for each testfile with same name as the test file + .txt. Programs can use their 
% working directory if they need to keep temporary cache files or internal debugging info. Standard 
% output and standard error will be logged.



%% output should be :
% start_time end_time chord_label
% times in seconds, chord labels corresponding to the syntax described by C. Harte et al. (2005)
filelist = importdata(trainlistfile);
testfiles = cell(size(filelist));

parfor i=1:size(filelist,1)
    %% get testfiles
    %% convert to features
    disp(filelist{i});
    testfiles{i} = createFeatures2NoGT(filelist{i},scratchdir);
end


load(strcat(scratchdir,'\model.mat'));

%% test program
testSAEHMM2(nn,inits,transitions,testfiles,resultdir)
%% create output



end