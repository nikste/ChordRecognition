function [gts_train,ffts_train] = decimateTrain(ffts_train,gts_train,chords_present,decimation_factor)
% only reduces chords that are plenty in the dataset.

decimated = 0;

rows_to_remove = [];
for ind=1:decimation_factor:size(gts_train,1)
    %disp(strcat(num2str(ind),' from ',num2str(size(gts_train,1))));
    
    if(chords_present(gts_train(ind)+1) > 150)
        chords_present(gts_train(ind)+1) = chords_present(gts_train(ind)+1) - 1;

        rows_to_remove = [rows_to_remove;ind];
    else
        decimated = decimated + 1;
    end
end
ffts_train(rows_to_remove,:) = [];
gts_train(rows_to_remove,:) = [];

end