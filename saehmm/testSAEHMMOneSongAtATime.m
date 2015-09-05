function [error] = testSAEHMMOneSongAtATime(nn,inits,transitions,ffts_test_no_preprocessing,gt_test,gt_test_lab,multi_res)%,M,P,COEFF,m)

%% do whitening stuff like we did before
disp('transforming input')
tic
% right_total = 0;
% wrong_total = 0;

Bs = {};
labels = {};
labels_chunks = {};
for song = 1:length(gt_test)

    % pca: BAD, your results are bad and you should feel bad!
%     ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song}));
%     ffts_test_no_preprocessing{song} = bsxfun(@minus,ffts_test_no_preprocessing{song},m);%mean(ffts_test_no_preprocessing{song}));
%     ffts_test_no_preprocessing{song} = normr(ffts_test_no_preprocessing{song}*COEFF);
%     f = ffts_test_no_preprocessing{song};
%     ffts_test_no_preprocessing{song} = f(:,1:1200);

    %multires
    if (multi_res == 1)
        ffts_test_no_preprocessing{song} = createMultiResolutionFFT(ffts_test_no_preprocessing{song});
    else
        ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song}));
    end

    %till now best performing:
    %ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song})); 
   
    % also not so good:
    %ffts_test_no_preprocessing{song} = normr(sqrt(ffts_test_no_preprocessing{song}));

% its okay   
%     ffts_test_no_preprocessing{song} = sqrt(ffts_test_no_preprocessing{song});
%     W =bsxfun(@minus,ffts_test_no_preprocessing{song},M) * P;%ffts_test_no_preprocessing{song}; 
%     ffts_test_no_preprocessing{song} = normr(W);

%performs bad
%     ffts_test_no_preprocessing{song} = preprocessData(ffts_test_no_preprocessing{song})



    song_data = ffts_test_no_preprocessing{song};
    
    [labels_1K,labels_song] = runNN(nn, song_data);

    labels{song} = labels_song;
    Bs{song} = labels_1K;
    right = 0;
    wrong = 0;
    [right,wrong] = computeError(labels(song),gt_test(song));
    disp(strcat('song:',num2str(song),'  we got : ',num2str(right/(right+wrong)*100),' % correct!'))
    label_chunk = chunkifyPrediction(labels{song});
    labels_chunks{song} = label_chunk;
    res = chordSymbolRecall(gt_test_lab{song},label_chunk);
    disp(strcat('Chord Symbol Recall:',num2str(res*100),' % correct!'))
    
    
    
    
    B = Bs{song};
    B = B';
    path = viterbi_path(inits,transitions,B) - 1;%conversion from state (is 1..25) to chord (is 0..24)
    paths{song} = path';
    
    right = 0;
    wrong = 0;
    [right,wrong] = computeError(paths(song),gt_test(song));
    disp(strcat('song:',num2str(song),'  we got : ',num2str(right/(right+wrong)*100),' % correct!'))
    path_chunk = chunkifyPrediction(paths{song});
    res = chordSymbolRecall(gt_test_lab{song},path_chunk);
    paths_chunks{song} = path_chunk;
    disp(strcat('Chord Symbol Recall:',num2str(res*100),' % correct!'))
    % free memory
    ffts_test_no_preprocessing{song} = [];
end
toc
[right_withoutSmoothing,wrong_withoutSmoothing] = computeError(labels,gt_test);
disp(strcat('only neural net : right=',num2str(right_withoutSmoothing),' wrong:',num2str(wrong_withoutSmoothing),' thats:',num2str(right_withoutSmoothing/(wrong_withoutSmoothing+right_withoutSmoothing) * 100),' % !!'))

res= weightedChordSymbolRecall(gt_test_lab,labels_chunks);
disp(strcat('WCSR:',num2str(res* 100),' % !!'))

% compute error (frame wise)
[right,wrong] =  computeError(paths,gt_test);

disp(strcat('final answer: right=',num2str(right),' wrong:',num2str(wrong),' thats:',num2str(right/(right+wrong) * 100),' % !!'))

% %weighted chord Symbol recall
res= weightedChordSymbolRecall(gt_test_lab,paths_chunks);
disp(strcat('WCSR:',num2str(res* 100),' % !!'))





% output result
error = res;%right / (right + wrong);
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
