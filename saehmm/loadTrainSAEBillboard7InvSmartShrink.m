function [ffts_train,gts_full,gts_train] = loadTrainSAEBillboard7InvSmartShrink(from,to,shrinkfactor,multi_res,id)
ffts_train = [];
gts_train = [];

%% load blacklist,
blacklistfile = strcat('/home/nsteen/billboard/data/songs/blacklist_',id);
blacklist = importdata(blacklistfile);

%% load all files not in blacklist

% load all filenames from gt folder (this does not need to be optimized)
gtfolder = '/home/nsteen/billboard/data/songs/gt7Inv';
gtfiles_all = myls(strcat(gtfolder,'/*.dataC'));

trainfiles = {};
blacklist2 = {};
%convert to strings
for i=1:length(blacklist)
    blacklist2{i}=  num2str(blacklist(i));
end
blacklist = blacklist2';
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

%% load gt files, do hmm counting (aka supervised training)
disp('loading ground truth')

if(to >= length(trainfiles))
    to = length(trainfiles);
end
if(from >= length(trainfiles))
    from = length(trainfiles);
end

gtfoldername = '/home/nsteen/billboard/data/songs/gt7Inv';
gts_full = {}; %for hmm training
gts_train = []; % for SAE training
for ind = from:to %length(trainfiles)
    
    %%disp(strcat(gtfoldername,'/',trainfiles{ind},'.dataC'))
    gts_loaded = importdata(strcat(gtfoldername,'/',trainfiles{ind},'.dataC'));
    gts_full{ind}  = gts_loaded(:,1);

end

%% shrink data for every loaded file for sae
disp('loading ffts')

ffts_train = [];
fftfoldername = '/home/nsteen/billboard/data/songs/fft';
indices = [];

parfor ind = from:to%length(trainfiles)

        disp(strcat('loading file:',num2str(ind)));
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
        d = size(gts_full{ind},1) - size(ffts_loaded,1);
        if(d ~= 0)
            disp(strcat('error in file:',trainfiles{ind},' difference:',num2str(size(gts_full{ind},1) - size(ffts_loaded,1))));
            if(d > 0) % add frame
                for i=1:d
                    ffts_loaded = [ffts_loaded;ffts_loaded(end,:)];
                end
            else
                ffts_loaded = ffts_loaded(1:end+d,:);
            end
            ffts_train = [ffts_train;ffts_loaded];
        else
            indices = [indices ind];
            ffts_train = [ffts_train;ffts_loaded];
        end

            %disp(strcat('loaded:',num2str(ind/length(trainfiles)*100),'%'));
        gts_train = [gts_train;gts_full{ind}];
        
        d = size(gts_full{ind},1) - size(ffts_loaded,1);
        disp(strcat('gts_full:',num2str(size(gts_full{ind},1)),':',num2str(size(ffts_loaded,1))));
        assert(d == 0);

end
[chords_present] =  testForAllChords(gts_train)
[gts_train,ffts_train] = decimateTrain(ffts_train,gts_train,chords_present,2);
[chords_present] =  testForAllChords(gts_train)
[gts_train,ffts_train] = decimateTrain(ffts_train,gts_train,chords_present,2);
[chords_present] =  testForAllChords(gts_train)
[gts_train,ffts_train] = decimateTrain(ffts_train,gts_train,chords_present,2);
[chords_present] =  testForAllChords(gts_train)
[gts_train,ffts_train] = decimateTrain(ffts_train,gts_train,chords_present,2);
[chords_present] =  testForAllChords(gts_train)
[gts_train,ffts_train] = decimateTrain(ffts_train,gts_train,chords_present,2);
[chords_present] =  testForAllChords(gts_train)

end
