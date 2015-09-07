function [] = createGT2(input_file,output_folder)
% wavfolder = '/home/nsteen/billboard/wav';
% labfolder = '/home/nsteen/billboard/McGill-Billboard';
% 
% gtfolder = '/home/nsteen/billboard/data/songs/gt7Inv';

% '/0000/majmin.lab'

% 
% wavfiles = myls(strcat(wavfolder,'/*.wav'));

stepsize = 1024;
samplingrate = 11025;
% for fileindex = 1:length(wavfiles)
%     
%     [pathstr,name,ext] = fileparts(wavfiles{fileindex});

labfilename = input_file;%strcat(labfolder,'/',name,'/majmin7inv.lab');
disp(labfilename);
% compute, remember the framerate is now 11025/s
[pathstr,name,ext] = fileparts(input_file)
n  = strcat(pathstr,'\',name)
sounddata = wavread(n);%wavfiles{fileindex});
sounddata = resample(sounddata,1,4);
%% get ground truth data in frames
len  = length(sounddata);

chordframes = createGT(labfilename,len,stepsize,samplingrate);

    %% save all that shit
disp(strcat('writing to file ' , name));

% ground truth
dlmwrite(strcat(output_folder,'\',name,'.dataC'),chordframes,' ');
% end

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
        aux = strread(labdata{l},'%s','delimiter','\t');
        for j=1:3
            
            labdescription(l,j) = aux(j);
            labdescription_chord(l,j) = aux(j);
            if(j == 3)
                %aux(j)
                %aux{j}
                labdescription_chord(l,j) = {string2chord7Inv(aux{j})};
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
    disp(strcat('too many frames:',num2str(toomanyframescounter)));
    disp(strcat('timedifference songs:',num2str(timedifferencesongs)));
end
