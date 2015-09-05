function [gt_test,ffts_test_no_preprocessing] = loadTestSAEHMM(id)
%[ffts_test,gt_test,ffts_test_no_preprocessing]

disp('loading files')

%% load blacklist,
blacklistfile = strcat('/home/nsteen/beatles_new/data/songs/blacklist_',id);
blacklist = importdata(blacklistfile);

%% load test set
% load complete files ffts
fftfolder = '/home/nsteen/beatles_new/data/songs2/fft';

ffts_test_no_preprocessing = {};

gtfolder = '/home/nsteen/beatles_new/data/songs2/gt';
gt_test = {};
shrinkfactor = 1;
%ffts_test = {};
for ind = 1:length(blacklist)
%     if(ind == floor(length(blacklist)/2))
%         disp(num2str(ind/length(blacklist)*100));
%     end
    %%disp(strcat('loading:',num2str(ind/length(blacklist)*100),' % : ',fftfolder,'/',blacklist{ind},'.dataF'))
    %ffts_test{ind} = preprocessData(importdata(strcat(fftfolder,'/',blacklist{ind},'.dataF')));
    ffts_test_no_preprocessing{ind} = importdata(strcat(fftfolder,'/',blacklist{ind},'.dataF'));
    
        gt_t = importdata(strcat(gtfolder,'/',blacklist{ind},'.dataC'));
    gt_test{ind} = gt_t(:,1);
    
        gts_full = gt_test;
        ffts_loaded = ffts_test_no_preprocessing{ind};
    
            d = size(gts_full{ind},1) - size(ffts_loaded,1);
        if(d ~= 0)
            disp(strcat('error in file:',' difference:',num2str(size(gts_full{ind}(1:shrinkfactor:end,:),1) - size(ffts_loaded(1:shrinkfactor:end,:),1))));
            if(d > 0) % add frame
                for i=1:d
                    ffts_loaded = [ffts_loaded;ffts_loaded(end,:)];
                end
            else
                ffts_loaded = ffts_loaded(1:end+d,:);
            end
            %ffts_train = [ffts_train;ffts_loaded(1:shrinkfactor:end,:)];
        else
            indices = [indices ind];
            %ffts_train = [ffts_train;ffts_loaded(1:shrinkfactor:end,:)];
        end

            %disp(strcat('loaded:',num2str(ind/length(trainfiles)*100),'%'));
        
        
        d = size(gts_full{ind}(1:shrinkfactor:end,:),1) - size(ffts_loaded(1:shrinkfactor:end,:),1);
        disp(strcat('gts_full:',num2str(size(gts_full{ind}(1:shrinkfactor:end,:),1)),':',num2str(size(ffts_loaded(1:shrinkfactor:end,:),1))));
        assert(d == 0);
        gt_train{ind} = gts_full{ind};
        ffts_test_no_preprocessing{ind} = ffts_loaded;
end
%ffts_test = preprocessDataCell(ffts_test);

% % load gt files
% gtfolder = '/home/nsteen/beatles_new/data/songs2/gt';
% 
% gt_test = {};
% for ind = 1:length(blacklist)
%     %%disp(strcat('loading:',gtfolder,'/',blacklist{ind},'.dataC')) 
%     gt_t = importdata(strcat(gtfolder,'/',blacklist{ind},'.dataC'));
%     gt_test{ind} = gt_t(:,1);
% %    size(gt_t)
% %    size(ffts_test{ind})
% %    assert(size(gt_t,1) == size(ffts_test{ind},1)); %assertion Failed, fft_test is 1 bigger?!
%     
% end



end