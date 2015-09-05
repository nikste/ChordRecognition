function  renameFilesHPABeatles( )
% renames files gotten from hpa tool, only use .entire

folder = '/home/nsteen/beatles_new/data/songs/hpa';

% load all filenames
filenames = myls(strcat(folder,'/*.entire'));


% go through all files,
for f = 1:length(filenames)

    disp(strcat('processing file:',num2str(f),' : ',filenames{f}));
   
    % open
    data = importdata(filenames{f});
    
    wavfilenamepath = data.textdata;
    wavfilenamepath = wavfilenamepath{1};
    data = data.data;
    
    [pathstr,name,ext] = fileparts(wavfilenamepath);
    
    % change to naming convention beatles (albumname__songname)
    idx = max(strfind(pathstr,'/'));
    name = strcat(pathstr(idx+1:end),'__',name);
    
    
    % cut off first two elements (time) of each subsequent line
    data = data(:,3:end);
    
    % save with different name
    savename = strcat(folder,'/',name,'.dataP');
    
    
    %should be folder/name.dataF
    
    dlmwrite(savename,data,' ');
    
    
    
end