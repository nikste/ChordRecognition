function [train_x,train_y] = shrinkDBNTrainData(train_x,train_y,shrinkfactor)


train_x = train_x(1:shrinkfactor:end,:);
train_y = train_y(1:shrinkfactor:end,:);

% dummy_x = [];
% dummy_y = [];
% length(train_x)
% % tic
% % for j=1:length(train_x)
% %     if(mod(j,10000)==0)
% %         toc
% %         j
% %         tic
% %     end
% %     if(mod(j,shrinkfactor)==0)
% %         dummy_x = [dummy_x;train_x(j,:)];
% %         dummy_y = [dummy_y;train_y(j,:)];
% %     else
% %     end
% % end
% % toc
% train_x = dummy_x;
% train_y = dummy_y;a







end