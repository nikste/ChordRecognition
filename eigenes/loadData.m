function outputlist = loadData()
foldername = '/home/nsteen/beatles/train/wav/';
filenames = dir(foldername);

foldernameTrain = '/home/nsteen/beatles/train/train/'
outputlist = []
for i=3:4%length(filenames)
    f = strcat(foldernameTrain,filenames(i).name);
    [pathstr,name,ext] = fileparts(f);
    
    pcpname = strcat(pathstr,'/',name,'.dataP');
    chordname = strcat(pathstr,'/',name,'.dataC');
    
    loadedPCP = load(pcpname);

    loadedChords = load(chordname);
    
    outputlist.append([loadedPCP,loadedChords]);
end





end