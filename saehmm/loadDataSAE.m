function [ffts_train,gts_full,gts_train] = loadDataSAE(shrinkfactor)

%% load blacklist,
blacklistfile = '/home/nsteen/beatles_new/data/songs/blacklist';
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

%% shrink data for every loaded file for sae
disp('loading ffts')

ffts_train = [];
fftfoldername = '/home/nsteen/beatles_new/data/songs/fft';

for ind = 1:length(trainfiles)
    tic
    %load file
    disp(strcat(fftfoldername,'/',trainfiles{ind},'.dataF'))
    ffts_loaded = importdata(strcat(fftfoldername,'/',trainfiles{ind},'.dataF'));
    %subsample
    ffts_train = [ffts_train;ffts_loaded(1:shrinkfactor:end,:)];
    toc
end

%% load gt files, do hmm counting (aka supervised training)
disp('loading ground truth')

gtfoldername = '/home/nsteen/beatles_new/data/songs/gt';
gts_full = {}; %for hmm training
gts_train = []; % for SAE training
for ind = 1:length(trainfiles)
    
    disp(strcat(gtfoldername,'/',trainfiles{ind},'.dataC'))
    gts_loaded = importdata(strcat(gtfoldername,'/',trainfiles{ind},'.dataC'));
    gts_full{ind}  = gts_loaded(:,1);
    gts_train = [gts_train;gts_loaded(1:shrinkfactor:end,1)];
end

end