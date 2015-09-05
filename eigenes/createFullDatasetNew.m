function createFullDatasetNew(albumname)
fft_feature_size = 3000;
%construct filenames for input
wavfolder = strcat('/home/nsteen/beatles_new/wav_aligned/The Beatles','/',albumname);
labfolder = strcat('/home/nsteen/beatles_new/annotations/chordlab/The Beatles','/',albumname);

%construct filenames for output
pcpfolder = '/home/nsteen/beatles_new/data/songs/pcp';
fftfolder = '/home/nsteen/beatles_new/data/songs/fft';
gtfolder = '/home/nsteen/beatles_new/data/songs/gt';

cqtfolder = '/home/nsteen/beatles_new/data/songs/cqt';
cqtlargefolder = '/home/nsteen/beatles_new/data/songs/cqt_large';


%precompute cqt filter:
minFreq =  32,703; 
maxFreq = 1975,53; 
fs = 11025;
bins = 12;
sparKernel = sparseKernel(minFreq,maxFreq,bins,fs,0.00005);
Q= 1/(2^(1/bins)-1);
fftLen= 2^nextpow2( ceil(Q*fs/minFreq) );
stepsize = 1024;

% bins = 36;
% sparKernel_large = sparseKernel(minFreq,maxFreq,bins,fs,0.00005);



labfiles = myls(strcat(labfolder,'/*.lab'));
wav_files = myls(strcat(wavfolder,'/*.wav'));


%check both input files
%assert(length(labfiles) == length(wavfiles))

%go through all songs
for fileindex = 1:length(wav_files)
    % check last 5 letters
    lastpartlab = labfiles{fileindex};
    lastpartlab = lastpartlab(end-10:end-3);
    lastpartwav = wav_files{fileindex};
    lastpartwav = lastpartwav(end-10:end-3);

    assert(strcmp(lastpartwav,lastpartlab))
    
    % construct output name
    [pathstr,name,ext] = fileparts(wav_files{fileindex});
    name = strcat(albumname,'__',name)
    
    
    % compute, remember the framerate is now 11025/s
    sound_data = wavread(wav_files{fileindex});
    
    %% get frames:cqt,pcp,fft
    % add margins
    sounddata_margins = [sound_data(1:fftLen/2,1);sound_data; sound_data(end - (fftLen)/2 + 1:end,1)];
    
    len = length(sounddata_margins);
    [ffts,cqts,pcps] = processWav(sounddata_margins,len,sparKernel,fftLen,stepsize,fft_feature_size);

    %% save all that shit
    disp(strcat('writing to file number:',num2str(fileindex),' of ' , num2str(length(wav_files)), ' name: ',wav_files{fileindex}))
    
    % check all
    assert(size(ffts,1) == size(cqts,1) );
    assert(size(ffts,1) == size(pcps,1));
    
    
    
    % ground truth
    %dlmwrite(strcat(gtfolder,'/',name,'.dataC'),chordframes,' ');
    
    % ffts
    %dlmwrite(strcat(fftfolder,'/',name,'.dataF'),ffts,' ');
    
    % cqts_large
    %dlmwrite(strcat(cqtlargefolder,'/',name,'.dataQL'),cqts_large,' ');
    
    % cqts
    dlmwrite(strcat(cqtfolder,'/',name,'.dataQ'),cqts,' ');
    
    % pcps
    dlmwrite(strcat(pcpfolder,'/',name,'.dataP'),pcps,' ');
    
end

end




%%%
% gets frames of ground truth according to sample size etc.
% labfilename : name of the labfile for current song
% len : length of the song in frames
% stepsize : stepsize of the fft
% samplingrate : samplingrate of the song

function chordframes = createGT(labfilename,len,stepsize,samplingrate)
    %%%load ground truth data from lab format
    labdata = importdata(labfilename);

    labdescription = cell(length(labdata),3);
    labdescription_chord = cell(length(labdata),3);
    
    for l = 1:length(labdata)
        aux = strread(labdata{l},'%s','delimiter',' ');
        for j=1:3
            
            labdescription(l,j) = aux(j);
            labdescription_chord(l,j) = aux(j);
            if(j == 3)
                %aux(j)
                %aux{j}
                labdescription_chord(l,j) = {string2chord(aux{j})};
                if(labdescription_chord{l,j} == -1)
                    warning('chord = -1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!')
                    warning(aux{j})
                end
            end
        end
    end
    
    
    %then we sample according to stepsize and length of the song
    lablinecounter = 1;
    toomanyframescounter  = 0;
    outputcounter = 0;
    
    %for statistical purposes
    timediffsongs = 0;
    
    for ind = 0:stepsize:len
        outputcounter = outputcounter + 1;
        timecounter_sec = ind / samplingrate;
        if(timecounter_sec > str2double(labdescription_chord{lablinecounter,2}))
            %%%dirty hack or security..
            if(lablinecounter == length(labdescription_chord))
                toomanyframescounter = toomanyframescounter + 1;
                timedifferencesongs = timecounter_sec - len / samplingrate;
            else
                lablinecounter = lablinecounter + 1;
                timedifferencesongs = timecounter_sec - len / samplingrate;
            end
        end
        %now we can easily save the results
        
        %save results
        chordframes{outputcounter,1} = labdescription_chord{lablinecounter,3}; %chord
        chordframes{outputcounter,2} = timecounter_sec; %time
    end
    toomanyframescounter
    timedifferencesongs
end


%%%% processes wav file, creates pcp, cqt_large, cqt  and fft
%%%% TODO:cqt_large
% sounddata : loaded wav file
% sparKernel : kernel of constant Q transform
% fftLen : length of the fft transform
% stepsize : stepsize of fft analization

function [ffts,cqts,pcps] = processWav(sounddata,len,sparKernel,fftLen,stepsize,fft_feature_size)

%% initialize values
no_frames = 0;
for framectr = 1:stepsize:(len - fftLen)
    no_frames = no_frames + 1;
end

ffts = zeros(no_frames,fft_feature_size);
cqts = zeros(no_frames,72);
pcps = zeros(no_frames,12);


linecounter = 0;
for framectr = 1:stepsize:(len - fftLen)
    
    %still_to_go = framectr - (len - fftLen)
    
    linecounter = linecounter + 1;
    
    %% first step ffts
    fft_var = fft(sounddata(framectr:framectr + fftLen) ,size(sparKernel,1));
    % shorten fft
    fft_var = abs(fft_var(1:fft_feature_size,:));
    ffts(linecounter,:) = fft_var;
    

    
    %% constant Q transform
    %size(sounddata)
    %framectr
    %(framectr+fftLen)
    
    cqt = abs(constQ(sounddata(framectr:framectr+fftLen)',sparKernel));
    cqts(linecounter,:) = cqt;
    

    
    %% pcp
    
    % aggregate
    for base_bin = 1:12
        for bin = base_bin:12:72
            pcps(linecounter,base_bin) = pcps(linecounter,base_bin) + abs( cqt(bin) );
        end
    end
    if(sum(pcps(linecounter,:)) == 0)
       pcps(linecounter,:) = [1;1;1;1;1;1;1;1;1;1;1;1] / 12 ;
    else
       pcps(linecounter,:) = pcps(linecounter,:) / sum(pcps(linecounter,:));
    end
    

end

end