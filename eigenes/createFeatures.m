function  createFeatures( input_folder, output_folder , fs )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
input_folder
output_folder
fs


fft_feature_size = 1500;

pcp_folder = strcat(output_folder,'/pcp');
fft_folder = strcat(output_folder,'/fft');
cqt_folder = strcat(output_folder,'/cqt');


%% precompute cqt filter:
minFreq =  32,703; 
maxFreq = 1975,53; 
%fs = 11025;
bins = 12;
sparKernel = sparseKernel(minFreq,maxFreq,bins,fs,0.00005);
Q= 1/(2^(1/bins)-1);
fftLen= 2^nextpow2( ceil(Q*fs/minFreq) );
% stepsize at most 100 ms
stepsize = 2^(nextpow2(fs/10)-1) %1024;
disp(strcat('using: stepsize:',num2str(stepsize),' thats: ',num2str(stepsize/fs),' and fftLen:',num2str(fftLen)))


% TODO in case of wav files change this back!
wav_files = myls(strcat(input_folder,'/*.wav'));



% check folders
% input folder exists
assert(1 == isequal(exist(input_folder, 'dir'),7));

% ouput folder exists
assert(1 == isequal(exist(output_folder, 'dir'),7));

% if subfolder dont exist, create:
if(~isequal(exist(pcp_folder, 'dir'),7))
    disp('pcp folder does not exist, creating..')
    mkdir(pcp_folder);
end
if(~isequal(exist(fft_folder, 'dir'),7))
    disp('fft folder does not exist, creating..')    
    mkdir(fft_folder);
end
if(~isequal(exist(cqt_folder, 'dir'),7))
    disp('cqt folder does not exist, creating..')    
    mkdir(cqt_folder);
end

disp(strcat('starting to process ',num2str(length(wav_files)),' files'))

parfor fileindex = 1:length(wav_files)
    tic
    % construct output name
    [pathstr,name,ext] = fileparts(wav_files{fileindex});
    
    % TODO: in case of flac data change this back!!
    sound_data = wavread(wav_files{fileindex});
    %sound_data = flacread2(wav_files{fileindex});
    %sound_data = mean(sound_data,2);
    
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
    
    
    % ffts
    %assert ffts
    assert(isreal(ffts));
    assert(1 == all(all(ffts >= 0)))
    disp(strcat('size ffts:',num2str(size(ffts))));
    %%fft_fname = strcat(fft_folder,'/',name,'.dataF')
    %%fft_fileID = fopen(fft_fname,'wb');
    %%fwrite(fft_fileID,ffts,'double');
    %%fclose(fft_fileID);
    %dlmwrite(strcat(fft_folder,'/',name,'.dataF'),ffts,' ');
    parsave(strcat(fft_folder,'/',name,'.dataF'),ffts);
    % cqts
    % assert cqts real and greater than zero
    assert(isreal(cqts))
    assert(1 == all(all(cqts >= 0)));
    %%cqt_fname = strcat(cqt_folder,'/',name,'.dataQ')
    %%cqt_fileID = fopen(cqt_fname,'wb');
    %%fwrite(cqt_fileID,cqts,'double');
    %%fclose(cqt_fileID);
    %dlmwrite(strcat(cqt_folder,'/',name,'.dataQ'),cqts,' ');
    
    % pcps
    %real and greater than zero
    assert(isreal(pcps) );
    assert(1 == all(all(ffts >= 0)));
    %%pcp_fname = strcat(pcp_folder,'/',name,'.dataP')
    %%pcp_fileID = fopen(pcp_fname,'wb');
    %%fwrite(pcp_fileID,pcps,'double');
    %%fclose(pcp_fileID);
    %dlmwrite(strcat(pcp_folder,'/',name,'.dataP'),pcps,' ');

    toc
end

end
function parsave(fname, data)
    save(fname, 'data')
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
    % and normalize if zero set everything equal
    if(sum(pcps(linecounter,:)) == 0)
       pcps(linecounter,:) = [1;1;1;1;1;1;1;1;1;1;1;1] / 12 ;
    else
       pcps(linecounter,:) = pcps(linecounter,:) / sum(pcps(linecounter,:));
    end

end

end


