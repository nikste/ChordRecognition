function executeOneCQ(basefolder)
minFreq =  32,703;%220;%27.5;
maxFreq = 1975,53;%880*2;
fs = 44100;
bins = 12;
sparKernel = sparseKernel(minFreq,maxFreq,bins,44100,0.00005);
Q= 1/(2^(1/bins)-1);
fftLen= 2^nextpow2( ceil(Q*fs/minFreq) );

%%find filenames
wavfolder= strcat(basefolder,'wav/');
listingwav = dir(wavfolder);

labfolder = strcat(basefolder,'lab/');
%sounddata = cell(length(listing));

trainfolder = strcat(basefolder,'train/');

% sum_error_pcp = 0;
% sum_error_pcp_mag = 0;
% sum_sq_error_pcp = 0;
% sum_sq_error_pcp_mag = 0;
matlabpool('open',6);
parfor i=3:length(listingwav)
    %'iteration'
    %i
    toomanyframescounter = 0;
    datestr(now)
    %for wave file load:
    wavfilename = strcat(wavfolder,listingwav(i).name)
    sounddata = wavread(wavfilename);
    
    ls = length(sounddata);

    sounddata = [sounddata(1:fftLen/2,1); sounddata; sounddata(ls - fftLen/2: ls,1)];
    
    %load lab for ground truth
    %prepare filename
    [pathstr,name,ext] = fileparts(wavfilename);
    labfilename = strcat(labfolder,name,'.lab')
    
    
    %%save filenames
    save_chordfilename = strcat(trainfolder,name,'.dataC');
    save_pcpfilename = strcat(trainfolder,name,'.dataP');
    save_cqfilename = strcat(trainfolder,name,'.dataQ');
    
    
    
    %%%load ground truth data from lab format
    labdata = importdata(labfilename);
    %perform cqt:
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
    labdescription;
    
    
    %%%perform cqt:
    
    %iterate through sounddata 
    timecounter_sec = 0;
    lablinecounter = 1;
    cqframes = [];
    pcpframes = [];
    chordframes = [];
    for f=1:2048:length(sounddata)-fftLen
        cqframe = constQ(sounddata(f:f+fftLen)',sparKernel);
        pcpframe = zeros(12);
        %pcpframe_mag = zeros(12);
        %compute naive pcp
        for base_bin = 1:12
            for bin = base_bin:12:72
                pcpframe(base_bin) = pcpframe(base_bin) +abs( cqframe(bin));
                %pcpframe_mag(base_bin )= pcpframe_mag(base_bin) + abs(cqframe(bin));
            end
        end
        %normalize pcp
        pcpframe = pcpframe/sum(pcpframe);
        %pcpframe_mag = pcpframe_mag / sum(pcpframe_mag);
        if(timecounter_sec > str2double(labdescription{lablinecounter,2}))
            %%%dirty hack or security..
            if(lablinecounter == length(labdescription))
                toomanyframescounter = toomanyframescounter + 1;
            else
                lablinecounter = lablinecounter + 1;
                
                %timecounter_sec
                %labdescription(length(labdescription),2)
                %labdescription{lablinecounter,3}
                %labdescription_chord{lablinecounter,3}

            end
        end
        cqframe = abs(cqframe);
        %labdescription
        %labdescription{lablinecounter,2}
        %lablinecounter
        timecounter_sec = f / 44100;
        %labdescription;
        
        
        %labdescription_chord{lablinecounter,3}
        %labdescription{lablinecounter,3}
        %%%%%%save everything
        cqframes = [cqframes cqframe'];
        pcpframes = [pcpframes pcpframe];

        chordframes = [chordframes labdescription_chord{lablinecounter,3}];
        
        %ppcp = perfectPcp(labdescription_chord{lablinecounter,3});
        
        %sum_error_pcp = sum_error_pcp + sum(abs(ppcp - pcpframe))
        %sum_error_pcp_mag = sum_error_pcp_mag + sum(abs(ppcp - pcpframe_mag))
        %sum_sq_error_pcp = sum_sq_error_pcp + sum((ppcp - pcpframe).^2)
        %sum_sq_error_pcp_mag = sum_sq_error_pcp_mag + sum((ppcp - pcpframe_mag).^2)
        
        %%bar([pcpframe ppcp]);
        %%tit = strcat(' ',num2str(timecounter_sec),' < ',labdescription{lablinecounter,2},' chord: ', labdescription{lablinecounter,3});
        %%title(tit);
        %%drawnow;
        %%pause(0.5)
    end
    
    %%% save this shit
    'writing to file'
    dlmwrite(save_chordfilename,chordframes', ' ');
    dlmwrite(save_pcpfilename,pcpframes',' ');
    dlmwrite(save_cqfilename,cqframes',' ');
    toomanyframescounter
    
end
matlabpool('close')
%x = wavread('/media/owner/Festpladde/uni/masterAI/MasterThesis/Datasets/beatles/AbbeyRoad/wav/1.wav');
%filler = zeros(32768/2);

%lab = importdata('/media/owner/Festpladde/uni/masterAI/MasterThesis/Datasets/beatles/AbbeyRoad/lab/1.lab');




end
