 For testing: 
% doChordID.sh "/path/to/testFileList.txt" "/path/to/scratch/dir" "/path/to/results/dir" 
% If there is no training, you can ignore the second argument here. In the results directory, there
% should be one file for each testfile with same name as the test file + .txt. Programs can use their 
% working directory if they need to keep temporary cache files or internal debugging info. Standard 
% output and standard error will be logged.



%% output should be :
% start_time end_time chord_label
% times in seconds, chord labels corresponding to the syntax described by C. Harte et al. (2005)
