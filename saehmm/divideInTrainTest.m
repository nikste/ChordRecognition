function [blacklist] = divideInTrainTest(offset,shrinkfactor,id)

blacklist = {};

%% load 
%%gtfolder = '/home/nsteen/beatles_new/data/songs/gt';
%%gtfiles = myls(strcat(gtfolder,'/*.dataC'));

% load all filenames from gt folder (this does not need to be optimized)
gtfolder = '/home/nikste/workspace-m/masterthesis/gt'
%gtfolder = '/home/nsteen/beatles_new/data/songs/gt';
% gtfiles_all = myls(strcat(gtfolder,'/*.dataC'));
% %remove fucking duplicates.
% if(length(gtfiles_all) ~= 180)
%    wd=gtfiles_all';
%    [~,idx]=unique(  wd , 'rows');
%    gtfiles_all=wd(idx,:)' ;
% end

% fuck myls
listing = dir(gtfolder)
listing = listing(3:end,:);
gtfiles = {};
for i=1:length(listing)
    gtfiles{i} = listing(i).name;
end

%% divide in train and test
%r = mod(rand(1) * 3,10); %random number between 1 and 10
% gtfiles(offset:4:end) = [];
blacklisted_cell = gtfiles(offset:shrinkfactor:end);
blacklisted = {};

% blacklisted = {};
% blacklisted_ctr = 1;
% for ind = 6:length(gtfiles)
%     if(mod(ind,5)==0)
%         blacklisted_cell{blacklisted_ctr} = gtfiles{ind};
%         disp(strcat('blacklisting:',gtfiles{ind}));
%         blacklisted_ctr = blacklisted_ctr + 1;
%     else
% 
%     end
% end
% disp(strcat('blacklisted:',num2str(blacklisted_ctr),' files'))

for ind = 1:length(blacklisted_cell)
    [pathstr,name,ext] = fileparts(blacklisted_cell{ind});
    blacklisted{ind} = name;
end

%% write blacklist to file
%blacklistfile = strcat('/home/nsteen/beatles_new/data/songs/blacklist_',id,'_',num2str(offset));
blacklistfile = strcat('/home/nikste/workspace-m/masterthesis/blacklist','_',num2str(offset));
fid = fopen(blacklistfile, 'w');

for ind = 1:length(blacklisted)
    fprintf(fid, '%s\n', blacklisted{ind});
end

fclose(fid);


%% return blacklist for files in test set
blacklist = blacklisted;
end
