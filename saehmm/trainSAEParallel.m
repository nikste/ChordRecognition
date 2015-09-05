function [ nns ] = trainSAEParallel( ffts_train,gts_train,gts_full, multi_res, pcp_pretrain )
%TRAINSAEPARALLEL Summary of this function goes here
%   Detailed explanation goes here
saes = {};
max_number = 66;
parfor i=1:max_number
    [l1,l2] = convertToLayers(i)
    
    % this is not the smartest way.
    if(l2 == 0)
        % dont deliver sae1 but save.
        [sae]= trainSAEHMMPre(0,ffts_train,gts_train,gts_full,l1,l2, multi_res, pcp_pretrain)
        saes{i} = {sae};
    else
        % deliver sae1 and save both.
        %[sae]= trainSAEHMMPre(,ffts_train,gts_train,gts_full,l1,l2, multi_res, pcp_pretrain)
        %nns{i} = {sae};
    end
    
end
for i=1:max_number
    [l1,l2] = convertToLayers(i)
    if(l2 == 0)
        % dont deliver sae1 but save.
        %[sae]= trainSAEHMMPre(0,ffts_train,gts_train,gts_full,l1,l2, multi_res, pcp_pretrain)
        %nns{i} = {sae};
    else
        %find sae with only one layer (last)
        x = i;
        l22 = 1;
        while(l22 ~= 0)
            [l21,l22] = convertToLayers(x)
            x = x-1;
        end
        x = x+1;
        % deliver sae1 and save both.
        [sae]= trainSAEHMMPre(saes{x}{1},ffts_train,gts_train,gts_full,l1,l2, multi_res, pcp_pretrain)
        saes{i} = {sae};
    end 
    
end




nns = {};
parfor i=1:max_number
        disp(strcat(num2str(i),'/'));
        [l1,l2] = convertToLayers(i)
        [nn,inits,transitions]= trainSAEHMMBackProp(saes{i}{1},ffts_train,gts_train,gts_full,l1,l2, multi_res, pcp_pretrain)
        nns{i} = {nn,inits,transitions};
end



end

function [l1,l2] = convertToLayers(x)
%takes number splits it, similar to hash

% compute boundaries:
boundaries = [1];
r = 1;
addition = 2;
while (r <= x)
    r = r + addition;
    addition = addition+1;
    boundaries = [boundaries r];
end

l1 = 1;
l2 = 0;
ctr = 1;
min = 1;
while (x  >= boundaries(ctr) )
    min = ctr;
    ctr = ctr + 1;
end
l1 = min;
l2 = x - boundaries(min);


l1 = l1*200;
l2 = l2*200;
end