function [ffts_train,gts_full,gts_train,ffts_train_songs] = loadTrainSAEJOINT(shrinkfactor,multi_res,id)

%% load blacklist,
blacklistfile = strcat('/home/nsteen/beatles_new/data/songs/blacklist_',id);
blacklist = importdata(blacklistfile);

%% load all files not in blacklist

% load all filenames from gt folder (this does not need to be optimized)
gtfolder = '/home/nsteen/beatles_new/data/songs/gt';
gtfiles_all = myls(strcat(gtfolder,'/*.dataC'));

trainfiles = {};

trainfilectr = 1;
for ind = 1:length(gtfiles_all)
    [pathstr,name,ext] = fileparts(gtfiles_all{ind});
    if(ismember(name,blacklist) == 0)
        trainfiles{trainfilectr} = name;
        trainfilectr = trainfilectr + 1;
    else
        disp(strcat('omitted:',name));
    end
end
disp(strcat('number of trainfiles:',num2str(length(trainfiles))));
%% shrink data for every loaded file for sae
disp('loading ffts')
%% load gt files, do hmm counting (aka supervised training)
disp('loading ground truth')

ffts_train = [];
ffts_train_songs = {};
fftfoldername = '/home/nsteen/beatles_new/data/songs/fft';

gtfoldername = '/home/nsteen/beatles_new/data/songs/gt';
gts_full = {}; %for hmm training
gts_train = []; % for SAE training
parfor ind = 1:length(trainfiles)
    %%tic
    %load file
    %%disp(strcat(fftfoldername,'/',trainfiles{ind},'.dataF'))
    ffts_loaded = importdata(strcat(fftfoldername,'/',trainfiles{ind},'.dataF'));
   
    %compute multi resolution ffts
    if (multi_res == 1)
        ffts_loaded = createMultiResolutionFFT(ffts_loaded);
    else
        ffts_loaded = normr(sqrt(ffts_loaded));
    end
    %subsample
    
    %this can be done smarter.
%     end_first_half = floor(size(ffts_loaded,1)/2);
%     start_second_half = floor( size(ffts_loaded,1)/2+1);
%     ffts_train = [ffts_train;ffts_loaded(1:2:end_first_half,:);ffts_loaded(start_second_half:4:end,:)];
    ffts_train_songs{ind} = ffts_loaded;
    ffts_train = [ffts_train;ffts_loaded(1:shrinkfactor:end,:)];
    
        %%disp(strcat(gtfoldername,'/',trainfiles{ind},'.dataC'))
    gts_loaded = importdata(strcat(gtfoldername,'/',trainfiles{ind},'.dataC'));
    gts_full{ind}  = gts_loaded(:,1);
    gts_train = [gts_train;gts_loaded(1:shrinkfactor:end,1)];
        %disp(strcat('loaded:',num2str(ind/length(trainfiles)*100),'%'));
    
    %%toc
end




end
