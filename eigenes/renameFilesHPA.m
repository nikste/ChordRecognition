function  renameFilesHPA( )
% renames files gotten from hpa tool, only use .entire

folder = '/home/nsteen/billboard/data/hpa';

% load all filenames
filenames = myls(strcat(folder,'/*.entire'));


% go through all files,
parfor f = 1:length(filenames)

    disp(strcat('processing file:',num2str(f)));
   
    % open
    data = importdata(filenames{f});
    
    wavfilenamepath = data.textdata;
    wavfilenamepath = wavfilenamepath{1};
    data = data.data;
    
    [pathstr,name,ext] = fileparts(wavfilenamepath);
    % cut off first two elements (time) of each subsequent line
    data = data(:,3:end);
    
    % save with different name
    savename = strcat(folder,'/',name,'.dataP');
    
    
    %should be folder/name.dataF
    
    dlmwrite(savename,data,' ');
    
    
    
end




end