function [ missed,madeup,correct ] = computeMissedMadeupCorrect( gt,est )
%computeMissedMadeupCorrect 
% gt is ground truth
% est is estimated chords from algorithm
% counts the number of missed chord types, made up chords that are not in
% the ground truth, and correctly classified chords by chord type.

% convert to logicals
gt_bin = gt > 0 ;
est_bin = est > 0 ;

res = gt_bin - est_bin;
% zero indicates correct,
% negative is made up
% positive is missed
missed = res > 0;
madeup = res < 0;

res = gt_bin + est_bin;
correct = res > 1;




end

