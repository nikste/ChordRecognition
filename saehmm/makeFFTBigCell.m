function [ ffts_test_big_no_preprocessing ] = makeFFTBigCell( ffts_test_no_preprocessing )
%MAKEFFTBIG concatenates 1500 fft to 4500, step before, step , step after
%   ffts_test_no_preprocessing: (frames, 1500) matrix of ffts


ffts_test_big_no_preprocessing = {};

for song = 1:size(ffts_test_no_preprocessing,2)
    disp(strcat('processing:',num2str(song),' of ',num2str(size(ffts_test_no_preprocessing,2))));
   ffts = ffts_test_no_preprocessing{song};
   ffts_big = zeros(size(ffts,1),4500);

%    size(ffts)
%    size(ffts_big)
%    
%    size([ffts(1,:) ffts(1,:) ffts(2,:)])
   
   %doubplicate first frame
   ffts_big(1,:) = [ffts(1,:) ffts(1,:) ffts(2,:)];
   for frame = 2:size(ffts,1)-1
        ffts_big(frame,:) = [ffts(frame-1,:) ffts(frame,:) ffts(frame+1,:)];
   end
   %douplicate last frame
   ffts_big(end,:) = [ffts(end-1,:) ffts(end,:) ffts(end,:)];
   ffts_test_big_no_preprocessing{song} = ffts_big;
end


end

