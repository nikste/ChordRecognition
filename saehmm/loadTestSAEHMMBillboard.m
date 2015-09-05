function [gt_test,ffts_test_no_preprocessing] = loadTestSAEHMMBillboard(id)
%[ffts_test,gt_test,ffts_test_no_preprocessing]

disp('loading files')

%% load blacklist,
blacklistfile = strcat('C:\stuff\masterthesis\blacklist_',id);%strcat('/media/nikste/moarspace/masterthesis/blacklist_',id);
blacklist = importdata(blacklistfile);
blacklist2 = {};
%convert to strings
%for i=1:length(blacklist)
%    blacklist2{i}=  num2str(blacklist(i));
%end
blacklist2 = convertToStrings(blacklist);


blacklist = blacklist2';
%% load test set
% load complete files ffts
fftfolder = 'C:\stuff\masterthesis\data\fft';%'/media/nikste/moarspace/masterthesis/data/fft';

ffts_test_no_preprocessing = {};
%ffts_test = {};
parfor ind = 1:length(blacklist)
    disp(strcat('opening file:',blacklist{ind}));
    %     if(ind == floor(length(blacklist)/2))
%         disp(num2str(ind/length(blacklist)*100));
%     end
    %%disp(strcat('loading:',num2str(ind/length(blacklist)*100),' % : ',fftfolder,'/',blacklist{ind},'.dataF'))
    %ffts_test{ind} = preprocessData(importdata(strcat(fftfolder,'/',blacklist{ind},'.dataF')));
    try
        ffts_test_no_preprocessing{ind} = load(strcat(fftfolder,'\',blacklist{ind},'.dataF'),'-mat');
        ffts_test_no_preprocessing{ind} = ffts_test_no_preprocessing{ind}.data;
    catch
        warning(['Problem loading fft file.', strcat(fftfolder,'\',blacklist{ind},'.dataF') '. Does it exist?']);
    end
end
%ffts_test = preprocessDataCell(ffts_test);

% load gt files
gtfolder = 'C:\stuff\masterthesis\gt';%'/media/nikste/moarspace/masterthesis/data/gt';

gt_test = {};
for ind = 1:length(blacklist)
    %%disp(strcat('loading:',gtfolder,'/',blacklist{ind},'.dataC')) 
    try
        gt_t = importdata(strcat(gtfolder,'\',blacklist{ind},'.dataC'));
        gt_test{ind} = gt_t(:,1);
    catch
        warning(['Problem loading ground truth file.', strcat(fftfolder,'\',blacklist{ind},'.dataF') '. Does it exist?']);
    end
end



end
