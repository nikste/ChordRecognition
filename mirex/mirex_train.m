function [filelist] = mirex_train(trainlistfile,scratchdir)

%% input should be:
% extractFeaturesAndTrain  "/path/to/trainFileList.txt"  "/path/to/scratch/dir"  
% where fileList.txt has the paths to each WAV file. The features extracted on this stage can be
% stored under /path/to/scratch/dir. The ground truth files for the supervised learning will be in 
% the same path with a .txt extension at the end. For example for /path/to/trainFile1.wav, there will
% be a corresponding ground truth file called /path/to/trainFile1.wav.txt.


%% load training files, compute features, get ground truth, align both.
filelist = importdata(trainlistfile)
trainfiles = []
gtfiles = []
parfor i=1:size(filelist,1)
    disp(filelist{i});
    [trainfile,gtfile] = createFeatures2(filelist{i},scratchdir);
    trainfiles = [trainfiles;trainfile];
    gtfiles = [gtfiles;gtfile];
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

%% train 
initTrain7Inv(trainfiles,gtfiles);


%% start training



%% save model




% downsample and compute features (fft 1500 bins)

%%filelist(~cellfun('isempty',filelist))  ;
% save('E:\stuff\repos\datasets\billboard\fl_b','filelist');
% dlmwrite('E:\stuff\repos\datasets\billboard\filelist2.txt',filelist,'');
% training

% 

end
