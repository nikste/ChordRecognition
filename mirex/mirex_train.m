function [filelist] = mirex_train(trainlistfile,scratchdir)

%% input should be:
% extractFeaturesAndTrain  "/path/to/trainFileList.txt"  "/path/to/scratch/dir"  
% where fileList.txt has the paths to each WAV file. The features extracted on this stage can be
% stored under /path/to/scratch/dir. The ground truth files for the supervised learning will be in 
% the same path with a .txt extension at the end. For example for /path/to/trainFile1.wav, there will
% be a corresponding ground truth file called /path/to/trainFile1.wav.txt.


%% load training files, compute features, get ground truth, align both.
filelist = importdata(trainlistfile);
trainfiles = cell(size(filelist));
gtfiles = cell(size(filelist));
parfor i=1:size(filelist,1)
    disp(filelist{i});
    [trainfile,gtfile] = createFeatures2(filelist{i},scratchdir);
    
    trainfiles{i} = trainfile;
    gtfiles{i} = gtfile;
    %[path,name,ext] = fileparts(filelist{i});
    %trainfiles{i} = strcat(scratchdir,'\fft\',name,'.dataF');
    %gtfiles{i} = strcat(scratchdir,'\gt\',name,'.dataC');
    
    %% check if length of files are good
    
    
%     try
%         [pathstr,name,ext] = fileparts(filelist{i}); 
%         base = 'E:\stuff\repos\datasets\billboard\McGill-Billboard';
%         labfile = strcat(base,'\',name,'\majmin7inv.lab');
%         if exist(labfile, 'file') == 2
%         else
%             filelist{i} = '';
%         end
%         destfile = strcat(filelist{i},'.txt');
%         disp([labfile ' ' destfile]);
%         copyfile(labfile,destfile);
%     catch exception
%         disp(exception)
%     end
        
end

%% train hmm
[inits,transitions] = countHMMParams(loadFiles(gtfiles));



%% train neural network
nn = initTrain7Inv(trainfiles,gtfiles,scratchdir);


%% save model

save(strcat(scratchdir,'\model'),'inits','transitions','nn');



% downsample and compute features (fft 1500 bins)

%%filelist(~cellfun('isempty',filelist))  ;
% save('E:\stuff\repos\datasets\billboard\fl_b','filelist');
% dlmwrite('E:\stuff\repos\datasets\billboard\filelist2.txt',filelist,'');
% training

% 

end


%% countHMMParams(gts_full)
% gets ground truth, computes the parameters of the HMM in a supervised
% manner
% gts_full : cell array of matrices(1,num_frames) for each song in training
% set
% outputs:
% inits : initial state probability of the hmm
% transitions : transition probability for hmm

function [inits,transitions] = countHMMParams(gts_full)
transitions = zeros(217,217);
inits = zeros(217);

%every song
for song = 1:length(gts_full)
    s = gts_full{song};
    % every frame in the song, everything is one dimensional, so all good
    previous = s(1) + 1; % add one for correct index for chord (matlab sigh)
    inits(previous) = inits(previous) + 1;
    for frame = 2:length(s)
        now = s(frame) + 1;
        transitions(previous,now) = transitions(previous,now) + 1;
        previous = now;
    end
end

% no ones left behind:
transitions = transitions + ones(size(transitions));
inits = inits + ones(size(inits));

% normalize transition TODO:carefull i'm tired, might not work correctly.
X = transitions';
X = X./( ones(size(X)) * diag(sum(abs(X))) );
transitions = X';
%normalize initial probability distribution
inits = inits / sum(inits);
end

function [train_y] = loadFiles(gtfilelist)
train_y = cell(size(gtfilelist));
for ind = 1 : size(gtfilelist,1)
    filename = gtfilelist{ind};
    train_y_ = importdata(filename);
    train_y_ = train_y_(:,1);
    train_y{ind} = train_y_;
    
end
end
