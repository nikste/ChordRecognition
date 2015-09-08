function [] = testSAEHMM2(nn,inits,transitions,testfiles,resultdir)

%% load training data
ffts = cell(size(testfiles));
for song = 1:size(testfiles,1)
    %filename = strcat(fftfoldername,'\',filelist_local{ind},'.dataF');
    %disp(['loading file:' filename ])
    filename = testfiles{song};
    train_x_ = load(filename,'-mat');
    ffts{song} = train_x_.data;
end


%% process all frames with pretrained neural network
Bs = {};
labels = {};
labels_chunks = {};
for song = 1:size(testfiles)    
    song_data = ffts{song};
    
    [labels_1K,labels_song] = runNN(nn, song_data);

    labels{song} = labels_song;
    Bs{song} = labels_1K;
    label_chunk = chunkifyPrediction(labels{song});
    labels_chunks{song} = label_chunk;
end


%% find most probable path with hmm
disp('hmm computes most probable path')

paths = {};
paths_chunks = {};
for song = 1:size(testfiles)
    disp(['song: ' num2str(song) ' of ' num2str(size(testfiles,1)) ]); 
    B = Bs{song};
    B = B';
    path = viterbi_path(inits,transitions,B) - 1;%conversion from state (is 1..25) to chord (is 0..24)
    paths{song} = path';
    
    path_chunk = chunkifyPrediction(paths{song});
    paths_chunks{song} = Num2Chord(path_chunk);
end


% check if output folder exists:
if(~isequal(exist(resultdir, 'dir'),7))
     disp('result folder does not exist, creating..')    
     mkdir(resultdir);
end
%% save results
for song = 1:size(testfiles)
    [path,name,ext] = fileparts(testfiles{song});
    dlmwrite(strcat(resultdir,'\',name,'.wav','.txt'),paths_chunks{song},'');

end

end


%% runNN
% runs the neural network with input
% computes the output in chord label and as 1 of K representation
function [res,label] = runNN(nn,x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    res = nn.a{end};
    [~, i] = max(nn.a{end},[],2);
    label = i - 1;%because chords.
end


%% computeError
% computes the error between two cells with matrices from labels
% prediction : predicted labels
% gt_test : ground truth of test data
% right : frames classified correctly
% wrong : frames classified wrong.
function [right,wrong] = computeError(prediction,gt_test)
assert(size(gt_test,1) == size(prediction,1));
right = 0;
wrong = 0;
for song = 1:length(gt_test)
    path = prediction{song};
    gt = gt_test{song};

    

    for frame = 1:size(path,1)
        if (path(frame) == gt(frame))
            %disp(strcat(num2str(gt(frame)),' : ',num2str(path(frame))))
            right = right + 1;
        else
            %disp(strcat(num2str(gt(frame)),' : ',num2str(path(frame)),' not quit right!'))
            wrong = wrong + 1;
        end
    end
end
end
