function [ gt_test_lab ] = loadLabTestBillboard7Inv( id)
%LOADLABTEST Summary of this function goes here
%   Detailed explanation goes here

labfolder = strcat('/home/nsteen/billboard/McGill-Billboard');


%% load blacklist,
blacklistfile = strcat('/home/nsteen/billboard/data/songs/blacklist_',id);
blacklist = importdata(blacklistfile);

% lab_album_filenames = {};
% for ind = 1:size(blacklist,1)
%    blacklist_filename = blacklist{ind};
%    s = strfind(blacklist{ind},'__');
%    st = strcat(blacklist_filename(1:s-1),'/',blacklist_filename(s+2:end));
%    lab_album_filenames{ind} = st;
% end

blacklist2 = {};
%convert to strings
%convert to strings
%for i=1:length(blacklist)
%    blacklist2{i}=  num2str(blacklist(i));
%end
blacklist2 = convertToStrings(blacklist);
blacklist = blacklist2';

% load gt files
gt_test_lab = {};
for ind = 1:length(blacklist)
    loadname = strcat(labfolder,'/',blacklist{ind},'/majmin7inv.lab');
    disp(strcat('loading:',loadname)) ;
    lab_raw = importdata(loadname);
    aux_mat = zeros(size(lab_raw,1),3);
    for j = 1:size(lab_raw,1)
        aux = strread(lab_raw{j},'%s','delimiter','\t');
        aux_mat(j,:) = [str2num(aux{1});str2num(aux{2});string2chord7Inv(aux{3})];
    end
    gt_test_lab{ind} = aux_mat;
end

end

