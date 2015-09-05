function [dummy_x,dummy_y] = loadDBNTrainFFTRaw(from,to)


foldername = '/home/nsteen/beatles/train/'
gtfoldername = strcat(foldername,'gt/')
datafoldername = strcat(foldername,'fft/')

filenames = dir(gtfoldername)

train_x = [];
train_y = [];
test_x = [];
test_y = [];

dummy_x = [];
dummy_y = [];
'loading files'
tic
number_of_files = length(filenames)
% from = 3
% to = number_of_files;
for j = from:to%71:length(filenames)%3:length(filenames)
    j
    [pathstr,name,ext] = fileparts(filenames(j).name) ;
    dataname = strcat(datafoldername,name,'.dataF');
    gtname =  strcat(gtfoldername,name,'.dataC');
    dummy_x = [dummy_x;importdata(dataname)];
    dummy_y = [dummy_y;importdata(gtname)];
end
% size_dummy_x = size(dummy_x)
% size_dummy_y = size(dummy_y)
% toc
% 
% 'converting to 1 of K'
% tic
% dummy_y = convertTo1K(dummy_y);
% toc
% 
% 
% 'creating test and trianing'
% %divide in train and test
% r = 0;%randi(10);
% %get size the stupid way
% te_list = [];
% tr_list = [];
% te_c = 0;
% tr_c = 0;
% for j = 1:length(dummy_y)
%     if(mod(j,10) == 0)
%         te_c = te_c +1;
%         te_list = [te_list te_c];
%     else
%         tr_c = tr_c + 1;
%         te_list = [te_list tr_c];
%     end
% end
% 
% test_x = zeros(te_c,215);
% test_y = zeros(te_c,25);
% train_x = zeros(tr_c,215);
% train_y = zeros(tr_c,25);
% 
% tic
% ct_test = 0;
% ct_train = 0;
% for j = 1:length(dummy_y)
%     if(mod(j,10) == 0)
%         count = j+r;
%         ct_test = ct_test + 1;
%         test_x(ct_test,:) = dummy_x(count,:);
%         test_y(ct_test,:) = dummy_y(count,:);
%     else
%         ct_train = ct_train+1;
%         train_x(ct_train,:) = dummy_x(j,:);
%         train_y(ct_train,:) = dummy_y(j,:);
%     end
% end
% 
% % for j = 1:length(dummy_y)
% %     if(mod(j,10) == 0)
% %         count = j+r;
% %         test_x(te_list(j),:) = dummy_x(count,:);
% %         test_y(te_list(j),:) = dummy_y(count,:);
% %     else
% %         train_x(te_list(j),:) = dummy_x(j,:);
% %         train_y(te_list(j),:) = dummy_y(j,:);
% %     end
% % end
% toc
end