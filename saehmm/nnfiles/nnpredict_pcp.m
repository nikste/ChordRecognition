function labels = nnpredict_pcp(nn, x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    %% old code
    %[~, i] = max(nn.a{end},[],2);
    %labels = i;
    
    %% pcp 3 index of three highest values
    % we assume the chance of having several values with the same hight is
    % neglible 
    [~, i] = sort(nn.a{end},2,'descend');
    labels = i(:,1:3);
end
