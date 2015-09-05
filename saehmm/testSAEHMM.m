function [error] = testSAEHMM(nn,inits,transitions,ffts_test_no_preprocessing,gt_test,gt_test_lab,multi_res)%,M,P,COEFF,m)

%% do whitening stuff like we did before
disp('transforming input')
tic
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
end
toc

%% process all frames with pretrained neural network
disp('neural network is computing features')

Bs = {};
labels = {};
labels_chunks = {};
sum_missed = zeros(217,1);
sum_madeup = zeros(217,1);
sum_correct = zeros(217,1);
for song = 1:length(gt_test)    
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
    
    %compute missed madeup and correct chords:
    est_chords = countChords7Inv( labels{song} );
    gt_chords = countChords7Inv( gt_test{song});
    [ missed,madeup,correct ] = computeMissedMadeupCorrect( gt_chords,est_chords );
    disp(strcat('missed:',num2str(sum(missed)),' made up:',num2str(sum(madeup)),' correct:',num2str(sum(correct))));
    %aggregate
    sum_missed = sum_missed + missed;
    sum_madeup = sum_madeup + madeup;
    sum_correct = sum_correct + correct;
end

right = 0;
wrong = 0;

[right,wrong] = computeError(labels,gt_test);

disp(strcat('only neural net : right=',num2str(right),' wrong:',num2str(wrong),' thats:',num2str(right/(right+wrong) * 100),' % !!'))
disp(strcat('sum_missed=',mat2str(sum_missed)));
disp(strcat('sum_madeup=',mat2str(sum_madeup)));
disp(strcat('sum_correct=',mat2str(sum_correct)));

res= weightedChordSymbolRecall(gt_test_lab,labels_chunks);
disp(strcat('WCSR:',num2str(res* 100),' % !!'))

%% find most probable path with hmm
disp('hmm computes most probable path')

paths = {};
paths_chunks = {};
sum_missed_hmm = zeros(217,0);
sum_madeup_hmm = zeros(217,0);
sum_correct_hmm = zeros(217,0);

for song = 1:length(gt_test)
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
    
        %compute missed madeup and correct chords:
    est_chords = countChords7Inv( paths{song} );
    gt_chords = countChords7Inv( gt_test{song});
    [ missed,madeup,correct ] = computeMissedMadeupCorrect( gt_chords,est_chords );
    disp(strcat('missed:',num2str(sum(missed)),' made up:',num2str(sum(madeup)),' correct:',num2str(sum(correct))));
    %aggregate
    sum_missed_hmm = sum_missed + missed;
    sum_madeup_hmm = sum_madeup + madeup;
    sum_correct_hmm = sum_correct + correct;
end

% compute error (frame wise)
[right,wrong] =  computeError(paths,gt_test);

disp(strcat('final answer: right=',num2str(right),' wrong:',num2str(wrong),' thats:',num2str(right/(right+wrong) * 100),' % !!'))
disp(strcat('sum_missed_hmm=',mat2str(sum_missed_hmm)));
disp(strcat('sum_madeup_hmm=',mat2str(sum_madeup_hmm)));
disp(strcat('sum_correct_hmm=',mat2str(sum_correct_hmm)));
% %weighted chord Symbol recall
res= weightedChordSymbolRecall(gt_test_lab,paths_chunks);
disp(strcat('WCSR:',num2str(res* 100),' % !!'))


% TODO: compute OR (for one song)
% TODO: compute WAOR (OR for all songs)

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
