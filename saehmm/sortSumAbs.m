function [w_sorted] = sortSumAbs(w)
%sorts neural network weights to get a nicer picture of stuff that it has
%learned.
[~,ind] = sort(sum(abs(w),2));
w_sorted = w(ind,:);



end