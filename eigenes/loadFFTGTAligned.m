function [ffts,gts] = loadFFTGTAligned(start,end_)

gts = [];
ffts = [];


% folder
fftfiles = myls('/home/nsteen/beatles_new/data/songs/fft/*.dataF');
gtfiles = myls('/home/nsteen/beatles_new/data/songs/gt/*.dataC');

% go through files

if(end_ >= length(gtfiles))
    will_set_end_to = length(gtfiles)
    end_ = length(gtfiles);
end

for f_ind = start:end_

    if (f_ind > end_)
        break;
    end
    
    disp(strcat('loading file:',num2str(f_ind),' of ',num2str(end_)));
    
    %% load gt file
    gt_data = importdata(gtfiles{f_ind});
    gt_data = gt_data(:,1); 
    
    gts = [gts; gt_data];
    
    %% load ffts
    fft_data = importdata(fftfiles{f_ind});
    ffts = [ffts;fft_data];
    
end

end