function [gt_test,ffts_test_no_preprocessing] = loadTestSAEHMMBillboard7Inv(id)
%[ffts_test,gt_test,ffts_test_no_preprocessing]

disp('loading files')

%% load blacklist,
blacklistfile = strcat('/home/nsteen/billboard/data/songs/blacklist_',id);
blacklist = importdata(blacklistfile);
blacklist2 = {};
%convert to strings
blacklist2 = convertToStrings(blacklist);
blacklist = blacklist2';
%% load test set
% load complete files ffts
fftfolder = '/home/nsteen/billboard/data/songs/fft';

ffts_test_no_preprocessing = {};
%ffts_test = {};
parfor ind = 1:length(blacklist)
%     if(ind == floor(length(blacklist)/2))
%         disp(num2str(ind/length(blacklist)*100));
%     end
    %%disp(strcat('loading:',num2str(ind/length(blacklist)*100),' % : ',fftfolder,'/',blacklist{ind},'.dataF'))
    %ffts_test{ind} = preprocessData(importdata(strcat(fftfolder,'/',blacklist{ind},'.dataF')));
    ffts_test_no_preprocessing{ind} = importdata(strcat(fftfolder,'/',blacklist{ind},'.dataF'));
end
%ffts_test = preprocessDataCell(ffts_test);

% load gt files
gtfolder = '/home/nsteen/billboard/data/songs/gt7Inv';

gt_test = {};
for ind = 1:length(blacklist)
    %%disp(strcat('loading:',gtfolder,'/',blacklist{ind},'.dataC')) 
    gt_t = importdata(strcat(gtfolder,'/',blacklist{ind},'.dataC'));
    gt_test{ind} = gt_t(:,1);
%    size(gt_t)
%    size(ffts_test{ind})
%    assert(size(gt_t,1) == size(ffts_test{ind},1)); %assertion Failed, fft_test is 1 bigger?!
    
end



end
