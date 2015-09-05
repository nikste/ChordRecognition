function [ ffts_b ] = makeFFTBig( ffts )

ffts_b = zeros(size(ffts,1),4500);

ffts_b(1,:) = [ffts(1,:) ffts(1,:) ffts(2,:)];
for frame = 2:size(ffts,1)-1
    if(mod(frame,1000)==0)
        disp(strcat('processing:',num2str(frame),' of ',num2str(size(ffts,1))));
    end
    ffts_b(frame,:) = [ffts(frame-1,:) ffts(frame,:) ffts(frame+1,:)];
end
%douplicate last frame
ffts_b(end,:) = [ffts(end-1,:) ffts(end,:) ffts(end,:)];



end