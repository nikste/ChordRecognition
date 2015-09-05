function testString2Chord(basefolder)
labfolder = strcat(basefolder,'lab/');
wavfolder= strcat(basefolder,'wav/');
listingwav = dir(wavfolder);
for i=3:length(listingwav)
    [pathstr,name,ext] = fileparts(wavfilename);
    labfilename = strcat(labfolder,name,'.lab')

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
                [aux(j) string2chord(aux{j})]
                
                labdescription_chord(l,j) = {string2chord(aux{j})};
                if(labdescription_chord{l,j} == -1|| labdescription_chord{l,j}>24)
                    warning('chord = -1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!')
                    warning(aux{j})
                end
            end
        end
    end
end



end