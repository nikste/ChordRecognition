function [ prediction_chunks ] = chunkifyPrediction( prediction )
%CHUNKIFYPREDICTION makes a "chunk" representation for chord symbol recall
% input :  prediction n,1 matrix, for each chord frame prediction
% output : prediciton_chunks matrix (frame|start_time,end_time,chord)

% initialize values
current_chord = prediction(1);
last_chord = prediction(1);

step_time = 1024/11025;
start_time = 0;
end_time = 0;
% go through all items
prediction_chunk_cell = {};
current_line = 1;

for lines = 2:length(prediction)
    current_chord = prediction(lines);
    % if item the same as the last do nothing
    
    % if item different from the last, add new line update start end    
    if (current_chord ~= last_chord)
        end_time = step_time * (lines - 1); 
        prediction_chunk_cell{current_line} = [start_time ; end_time; last_chord];
        assert(last_chord >= 0);


        % update to new values
        current_line = current_line + 1;
        start_time = end_time;
    end
    
    last_chord = current_chord;
    
    %if we are at the last chord, just save it without chord change
    if(lines == length(prediction))
        end_time = step_time * lines;
        prediction_chunk_cell{current_line} = [start_time ; end_time; last_chord];
        assert(last_chord >= 0);
    end
end

% special case only one element
if(length(prediction) == 1)
    prediction_chunk_cell{1} = [0.000; step_time; current_chord];
end
prediction_chunks = cell2mat(prediction_chunk_cell)';



end

