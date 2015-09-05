function [ffts_train] = createMultiResolutionFFT(ffts_train)
%% creates a multi resolution fft by averaging subsequent frames
% in hope that it will eradicate transient noise
% input: ffts_train a frames,1500 matrix (not preprocess)
% output: ffts_train a frames,4500 matrix preprocessed: 1-1500 immidiate
% frames 1500-3000 frames averaged over -1:0:1 and 3000-4500 frames
% averaged over -3,-2,-1,0,1,2,3



%create three vectors,-1;0;1 and -3;-2;-1;0;1;2;3
% disp('computing for -1:0:1')
disp('using 3 and 9 frames before and in front');
offset_first = 3;%1
offset_sec = 9;%3


ffts_one = zeros(size(ffts_train));
for frame = 1:offset_first%3
   ffts_two(frame,:) = median(ffts_train(1:offset_first,:));%frame+3,:));%median(ffts_train(1:frame+3,:)); 
end
for frame = (offset_first+1):size(ffts_one,1)-(offset_first+1)%4:size(ffts_one,1)-4
%     if(mod(frame,10000)==0)
%         disp(num2str(frame));
%     end
    ffts_one(frame,:) = median(ffts_train(frame-(offset_first):frame+(offset_first)));%median(ffts_train(frame-3:frame+3,:));%median(ffts_train(frame-1:frame+1,:));
end
for frame = size(ffts_train,1)-(offset_first+1):size(ffts_train,1)%size(ffts_train,1)-4:size(ffts_train,1)
   ffts_one(frame,:) = median(ffts_train(frame-offset_first:end,:));%median(ffts_train(frame-3:end,:)); %median(ffts_train(frame-3:end,:)); 
end
ffts_one = normr(sqrt(ffts_one));

% disp('computing for -3:3')

ffts_two = zeros(size(ffts_train));

for frame = 1:offset_sec%1:9
   ffts_two(frame,:) = median(ffts_train(1:frame+offset_sec,:));%median(ffts_train(1:frame+9,:));%median(ffts_train(1:frame+3,:)); 
end
for frame = (offset_sec+1):size(ffts_train,1)-(offset_sec+1)%10:size(ffts_train,1)-10
    %     if(mod(frame,10000)==0)
%         disp(num2str(frame));
%     end
   ffts_two(frame,:) = median(ffts_train(frame-offset_sec:frame+offset_sec));%median(ffts_train(frame-9:frame+9,:));%median(ffts_train(frame-3:frame+3,:));
end
for frame = size(ffts_train,1)-offset_sec:size(ffts_train,1)%size(ffts_train,1)-9:size(ffts_train,1)
   ffts_two(frame,:) = median(ffts_train(frame-offset_sec:end,:));%median(ffts_train(frame-9:end,:)); %median(ffts_train(frame-3:end,:)); 
end

% disp('compressing and normalizing:')

%normalize all
ffts_two = normr(sqrt(ffts_two));

ffts_train = normr(sqrt(ffts_train));

ffts_train = [ffts_train ffts_one ffts_two];
%%%%ffts_train = normr(sqrt(ffts_train));

end