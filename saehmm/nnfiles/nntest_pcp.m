function [er, bad] = nntest_pcp(nn, x, y)
    labels = nnpredict_pcp(nn, x);
    %% old code
    %[~, expected] = max(y,[],2);
    %bad = find(labels ~= expected);    
    %er = numel(bad) / size(x, 1);

    %% find the max 3 elements of labels and compare to max 3 of y
    % TODO: this is probably a shitty way to do this, but i did not see any
    % better way right now.
    
    % compare intersection of the two matrices y and labels (unsorted)
    
    %get sorted list of indices with highest values from targets
    [~,I] = sort(y,2,'descend');
    % only save first 3 elements
    I_max = I(:,1:3);
    
    bad = [];
    for i = 1:size(labels,1)
        
        
        %get intersection
        same = intersect(I_max(i,:),labels(i,:));
        
        %disp(strcat(' interpretation=',num2str(length(same) == 3),' pred=',mat2str(I_max(i,:)),' label=',mat2str(labels(i,:)),' intersect=',mat2str(same)));
        
        if(length(same) ~= 3) % if not all three max vectors match
            bad = [bad;1];
        end
        
    end
    er = numel(bad) / size(x, 1);

end
