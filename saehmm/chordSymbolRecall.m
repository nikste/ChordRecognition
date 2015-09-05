function res = chordSymbolRecall(labfile,predictions)
%%% calcultes the chord symbol recall as used in mirex 
% prediction, matrix(frames,3) [start_time, end_time, chord]
% ground truth, matrix(frames,3) [start_time, end_time, chord]


% % load gt and convert to matrix of pure numbers
% d = importdata('/home/nsteen/beatles_new/annotations/chordlab/The Beatles/01_-_Please_Please_Me/01_-_I_Saw_Her_Standing_There.lab')
% 
% f = {};
% 
% for i=1:length(d)
%     aux = strread(d{i},'%s','delimiter',' ')
%     f{i} = aux;
% end
% 
% % convert from chrods to numbers
% for i=1:length(f)
%     r = f{i};
%     r{3} = string2chord(r{3});
%     g(i,:) = [str2num(r{1}) str2num(r{2}) r{3}];
% end
% convert predictions to segments : cell{ind} = [start,end,chord]

% go through labfile segments

% compute overlap


time_start_gt_chunk = -1;
time_end_gt_chunk = -1;
label_gt = -1;

% we initialize with first chunk
time_start_pred_chunk = -1;
time_end_pred_chunk = -1;
label_pred = -1;


overlap = 0;
line_ctr_pred = 1;
line_ctr_gt = 1;
now = 0;

% s_p = size(predictions)
% s_g = size(labfile)

% jumping monkey is only happy when both trees have the same color
while(line_ctr_pred <= size(predictions,1) && line_ctr_gt <= size(labfile,1))
    time_end_pred_chunk = predictions(line_ctr_pred,2);
    time_end_gt_chunk = labfile(line_ctr_gt,2);
    label_pred = predictions(line_ctr_pred,3);
    label_gt = labfile(line_ctr_gt,3);
    
    %search for the next branch
    if(time_end_pred_chunk < time_end_gt_chunk)
        % pred closer
        if (label_gt == label_pred)
            %happy monkey
            overlap = overlap + (time_end_pred_chunk-now);
        end
        % move one branch up
        now = time_end_pred_chunk;
        line_ctr_pred = line_ctr_pred + 1;
    else
        % both same or gt closer
        if (label_gt == label_pred)
            % happy monkey
            overlap = overlap + (time_end_gt_chunk - now);
        end
        now = time_end_gt_chunk;
        line_ctr_gt = line_ctr_gt + 1;
    end
end




% for line = 1:size(labfile,1)
%     % first update gt pointers
%     time_start_gt_chunk = labfile(line,1);
%     time_end_gt_chunk = labfile(line,2);
%     label_gt = labfile(line,3);
% 
%     % check what segments i have in this range
%     % if it starts outside the range its over.
% 
%     while(time_start_pred_chunk < time_end_gt_chunk)
%         % check current chunk for first time its already initialized
%         % if label correct then add time interval
%         % to correctly classified:
%         
%         if(label_pred == label_gt)
%             % check the options we have, 
%             s = 0;
%             e = 0; 
%             % find the bigger value for start
%             if(time_start_pred_chunk < time_start_gt_chunk)
%                 s = time_start_gt_chunk;
%             else
%                 s = time_start_pred_chunk;
%             end
%             % find the smaller value for end
%             if(time_end_pred_chunk < time_end_gt_chunk)
%                 e = time_end_pred_chunk;
%             else
%                 e = time_end_gt_chunk;
%             end
%             
%             % substract end - start
%             overlap = overlap + (e - s);
%         end
%         
%         %move pointer of pred to next chunk
%         % break if its pred chunk is longer than gt chunk (end)
%         if (time_end_pred_chunk > time_end_gt_chunk )
%             break;
%         end
%         % if all analysed
%         if (line_ctr_pred == size(predictions,1))
%             break;
%         end
%         
%         line_ctr_pred = line_ctr_pred + 1;
%         % if does not match but lines are over
%         
%         time_start_pred_chunk = predictions(line_ctr_pred,1); 
%         time_end_pred_chunk =  predictions(line_ctr_pred,2);
%         label_pred = predictions(line_ctr_pred,3);
%     end
%     
%     % if all analysed
%     if (line_ctr_pred == size(predictions,1))
%         break;
%     end
% 
% end

% TODO: how to handle overlap if file is bigger than gt?
res = overlap/labfile(end,2);

end